# -*- coding: utf-8 -*-
"""
Standard Experimental Set Creator
- Input from Clean_Files sheet only (or optional multi-sheet for train)
- Sample exactly n_per_class per class
- Stratified 60/20/20 train/val/test split
- Fuzzy match missing files (=95%)
- Copy images + segmentations into split/class folders
- Save train/val/test Excel logs
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rapidfuzz import fuzz, process

# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================
# USER INPUT REQUIRED:
# Path to the tracking workbook generated earlier in the pipeline.
# The workbook is expected to contain a sheet named "Clean_Files" and, if
# CLEAN_FILES_ONLY is False, any additional sheets can also contribute to the
# training pool.
main_excel = "/path/to/AccRejRew_Clean_File_Tracking.xlsx"

# USER INPUT REQUIRED:
# Root directory containing image files organized by class subfolder:
#   images_root/
#       Accept/
#       Reject/
#       Rework/
# Segmentations may also be resolved relative to these folders when needed.
images_root = "/path/to/images_and_segmentations_root"

# USER INPUT REQUIRED:
# Parent directory where the curated standardized dataset will be created.
output_root = "/path/to/output_root"

# The script creates this dataset folder inside output_root.
output_set = os.path.join(output_root, "Standardized_Experimental_Set")

# Number of samples per class used in fully balanced mode. When
# BALANCE_VAL_TEST_ONLY = True, this value determines the per-class sizing of
# validation and test subsets (0.2 * n_per_class for each split).
n_per_class = 1000
random_seed = 42

# Mode switch:
# - False: fully balanced train/val/test split from Clean_Files only.
# - True:  balanced val/test from Clean_Files, fuller train set from remaining
#          rows (and optionally additional sheets).
BALANCE_VAL_TEST_ONLY = True

# If True, training samples come only from the Clean_Files sheet.
# If False, non-Clean_Files sheets can be appended to the training pool.
CLEAN_FILES_ONLY = False

# =============================================================================
# Load source data
# =============================================================================
# Clean_Files is always required because validation and test are derived from it.
df_clean = pd.read_excel(main_excel, sheet_name="Clean_Files")
df_clean = df_clean[df_clean["Str_Label"].isin(["Accept", "Reject", "Rework"])].copy()

# Optionally load all other sheets for use in the training pool.
if not CLEAN_FILES_ONLY:
    df_all = pd.read_excel(main_excel, sheet_name=None)
    other_sheets = {k: v for k, v in df_all.items() if k != "Clean_Files"}
    df_other = pd.concat(other_sheets.values(), ignore_index=True)
    df_other = df_other[df_other["Str_Label"].isin(["Accept", "Reject", "Rework"])].copy()

# Normalize image filenames so they consistently use the nnUNet-style image name
# convention with the "_0000.nii.gz" suffix.
for df_tmp in [df_clean] + ([df_other] if not CLEAN_FILES_ONLY else []):
    df_tmp["Names"] = df_tmp["Names"].apply(
        lambda x: x if x.endswith("_0000.nii.gz") else x.replace(".nii.gz", "_0000.nii.gz")
    )

classes = ["Accept", "Reject", "Rework"]

# =============================================================================
# Build train/val/test splits
# =============================================================================
if not BALANCE_VAL_TEST_ONLY:
    # Mode 1: fully balanced train/val/test using only Clean_Files rows.
    selected = []
    for cls in classes:
        cls_subset = df_clean[df_clean["Str_Label"] == cls]
        if len(cls_subset) < n_per_class:
            raise ValueError(
                f"Class '{cls}' only has {len(cls_subset)} samples, need {n_per_class}."
            )
        selected.append(cls_subset.sample(n=n_per_class, random_state=random_seed))

    selected_df = pd.concat(selected).reset_index(drop=True)

    # Standard 60/20/20 split.
    trainval_df, test_df = train_test_split(
        selected_df,
        test_size=0.20,
        stratify=selected_df["Str_Label"],
        random_state=random_seed
    )

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.25,
        stratify=trainval_df["Str_Label"],
        random_state=random_seed
    )

else:
    # Mode 2: balanced validation and test sets, fuller training set.
    val_test_blocks = []
    remaining_blocks = []

    for cls in classes:
        cls_subset = df_clean[df_clean["Str_Label"] == cls]

        # Each of val and test receives 0.2 * n_per_class samples per class.
        n_valtest_per_class = int(0.2 * n_per_class)

        if len(cls_subset) < n_valtest_per_class * 2:
            raise ValueError(
                f"Class '{cls}' has {len(cls_subset)} samples, "
                f"need at least {2 * n_valtest_per_class} for val+test."
            )

        sampled = cls_subset.sample(
            n=2 * n_valtest_per_class,
            random_state=random_seed
        )

        remaining = cls_subset.drop(sampled.index)

        val_test_blocks.append(sampled)
        remaining_blocks.append(remaining)

    val_test_df = pd.concat(val_test_blocks).reset_index(drop=True)

    # Split the sampled block equally into validation and test sets.
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.50,
        stratify=val_test_df["Str_Label"],
        random_state=random_seed
    )

    # Build the training dataframe from remaining Clean_Files rows and,
    # optionally, additional sheets.
    if CLEAN_FILES_ONLY:
        train_df = pd.concat(remaining_blocks).reset_index(drop=True)
    else:
        remaining_df = pd.concat(remaining_blocks).reset_index(drop=True)
        train_df = pd.concat([remaining_df, df_other]).reset_index(drop=True)

print("\n? Dataset composition:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# =============================================================================
# Add numeric labels
# =============================================================================
label_map = {"Accept": 0, "Reject": 1, "Rework": 2}
for df_tmp in [train_df, val_df, test_df]:
    df_tmp["Label"] = df_tmp["Str_Label"].map(label_map)

# =============================================================================
# Create output folder structure
# =============================================================================
splits = ["train", "val", "test"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_set, split, cls), exist_ok=True)

# =============================================================================
# File copy helpers
# =============================================================================
def fuzzy_find_missing(src_path, search_dir):
    """
    If file missing, fuzzy match inside search_dir.
    Returns best guess path or None.
    """
    files = os.listdir(search_dir)
    best = process.extractOne(
        os.path.basename(src_path), files, scorer=fuzz.partial_ratio
    )
    if best and best[1] >= 95:
        print(f"? Fuzzy matched: {os.path.basename(src_path)} ? {best[0]}")
        return os.path.join(search_dir, best[0])
    return None


def safe_copy(src, dst_dir):
    # Copy a file if present; otherwise attempt fuzzy recovery within the same
    # parent directory. Returns the copied output path or None.
    if os.path.exists(src):
        shutil.copy2(src, dst_dir)
        return os.path.join(dst_dir, os.path.basename(src))

    parent_dir = os.path.dirname(src)
    if os.path.isdir(parent_dir):
        match = fuzzy_find_missing(src, parent_dir)
        if match:
            shutil.copy2(match, dst_dir)
            return os.path.join(dst_dir, os.path.basename(match))

    print(f"? Missing file (no copy): {src}")
    return None


all_splits = {"train": train_df, "val": val_df, "test": test_df}

# =============================================================================
# Copy images and segmentations
# =============================================================================
print("\n?? Copying images + segmentations...")

for split_name, df_split in all_splits.items():

    out_dirs1, out_dirs2, out_seg_dirs = [], [], []

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Copying {split_name}"):

        fname       = row["Names"]
        cls         = row["Str_Label"]
        seg_src     = row.get("Seg_dirs", "")
        img_src     = os.path.join(images_root, cls, fname)

        dst_folder  = os.path.join(output_set, split_name, cls)

        # Copy the image file into the destination split/class folder.
        copied_img  = safe_copy(img_src, dst_folder)

        # Copy the segmentation file. If the absolute path in Seg_dirs is not
        # valid, fall back to resolving the basename relative to images_root.
        seg_name = os.path.basename(seg_src)
        seg_src_norm = seg_src if os.path.exists(seg_src) else os.path.join(images_root, cls, seg_name)
        copied_seg = safe_copy(seg_src_norm, dst_folder)

        # Duplicate the image path into two columns so downstream multi-image
        # pipelines can reuse the same image twice when needed.
        out_dirs1.append(copied_img)
        out_dirs2.append(copied_img)
        out_seg_dirs.append(copied_seg)

    df_split["Directories1"] = out_dirs1
    df_split["Directories2"] = out_dirs2
    df_split["Seg_Dirs"]     = out_seg_dirs

# =============================================================================
# Save split logs
# =============================================================================
train_df.to_excel(os.path.join(output_set, "AccRejRew_StandardExpSet_train_log.xlsx"), index=False)
val_df.to_excel(os.path.join(output_set, "AccRejRew_StandardExpSet_val_log.xlsx"), index=False)
test_df.to_excel(os.path.join(output_set, "AccRejRew_StandardExpSet_test_log.xlsx"), index=False)

print("\n? Standard Experimental Set complete!")
print(f"? Train samples: {len(train_df)}")
print(f"? Val samples:   {len(val_df)}")
print(f"? Test samples:  {len(test_df)}")
print(f"?? Output directory: {output_set}")
