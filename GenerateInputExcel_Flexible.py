import os
import pandas as pd
from sklearn.utils import shuffle

# =============================================================================
# Script purpose
# =============================================================================
# This script converts the train/val/test log Excel files from the standardized
# experimental set into model-input Excel files for downstream AutoQC training
# and evaluation.
#
# Expected input columns in each source workbook:
#   - Names
#   - Str_Label
#   - Seg_dirs
#   - Directories   or   Directories_1 / Directories_2
#
# The script supports three label-grouping modes:
#   - "problem1": Accept/Rework vs Reject
#   - "problem2": Accept vs Rework
#   - "multiclass": Accept vs Reject vs Rework
# =============================================================================

# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================
PROBLEM_MODE = "multiclass"  # Choose from: "problem1", "problem2", "multiclass"
INPUT_MODE = "multi"        # Choose from: "single", "multi"
USE_FULL_TRAIN = False       # Only affects training split selection logic for some problem modes

# USER INPUT REQUIRED:
# BASE_DIR must point to the directory that already contains the standardized
# train/val/test Excel logs created earlier in the pipeline.
BASE_DIR = "/path/to/Standardized_Experimental_Set"

# USER INPUT REQUIRED:
# OUTPUT_DIR is where the generated model-input Excel files will be written.
# Keeping OUTPUT_DIR = BASE_DIR is convenient when you want outputs saved next
# to the source split logs.
OUTPUT_DIR = BASE_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# These filenames are expected to exist inside BASE_DIR.
INPUT_FILES = {
    "train": os.path.join(BASE_DIR, "AccRejRew_StandardExpSet_train_log.xlsx"),
    "val":   os.path.join(BASE_DIR, "AccRejRew_StandardExpSet_val_log.xlsx"),
    "test":  os.path.join(BASE_DIR, "AccRejRew_StandardExpSet_test_log.xlsx"),
}

# Numeric label mappings used by the downstream training code.
LABEL_MAPS = {
    "problem1": {"Reject": 0, "Accept_Rework": 1},
    "problem2": {"Accept": 1, "Rework": 0},
    "multiclass": {"Accept": 0, "Reject": 1, "Rework": 2},
}

# =============================================================================
# Helper
# =============================================================================
def assign_directories(row, input_mode):
    # Standardize directory columns so downstream code sees the expected schema.
    # - single mode  -> output column: Directories
    # - multi mode   -> output columns: Directories_1, Directories_2
    #
    # If the source workbook only has a single directory column, that value is
    # duplicated when INPUT_MODE == "multi".
    if input_mode == "single":
        return {"Directories": row["Directories_1"] if "Directories_1" in row else row.get("Directories", "")}
    else:
        return {
            "Directories_1": row["Directories_1"] if "Directories_1" in row else row.get("Directories", ""),
            "Directories_2": row["Directories_2"] if "Directories_2" in row else row.get("Directories", "")
        }

# =============================================================================
# Process each split
# =============================================================================
for split in ["train", "val", "test"]:
    print(f"\n--- Processing {split} ({PROBLEM_MODE}) ---")

    # Read the split workbook and normalize string labels for consistent matching.
    df = pd.read_excel(INPUT_FILES[split])
    df["Str_Label"] = df["Str_Label"].astype(str).str.title()

    if split == "train":
        # Training split logic can optionally use all available training samples,
        # depending on PROBLEM_MODE and USE_FULL_TRAIN.
        if PROBLEM_MODE == "problem1":
            reject_df = df[df["Str_Label"] == "Reject"]
            acc_df = df[df["Str_Label"] == "Accept"]
            rew_df = df[df["Str_Label"] == "Rework"]

            if USE_FULL_TRAIN:
                # Keep all Reject + all Accept/Rework rows.
                df_proc = pd.concat([reject_df, acc_df, rew_df], ignore_index=True)
            else:
                # Keep all Reject rows and sample Accept/Rework to roughly match.
                n_reject = len(reject_df)
                acc_sample = acc_df.sample(n=min(n_reject // 2, len(acc_df)), random_state=42)
                rew_sample = rew_df.sample(n=min(n_reject - len(acc_sample), len(rew_df)), random_state=42)
                df_proc = pd.concat([reject_df, acc_sample, rew_sample], ignore_index=True)

            # Merge Accept and Rework into a single positive class.
            df_proc["Str_Label"] = df_proc["Str_Label"].replace({"Accept": "Accept_Rework", "Rework": "Accept_Rework"})

        elif PROBLEM_MODE == "problem2":
            # Keep only Accept and Rework rows.
            df_proc = df[df["Str_Label"].isin(["Accept", "Rework"])].copy()
        else:  # multiclass
            # Keep all three classes as-is.
            df_proc = df.copy()

    else:
        # Validation and test logic mirrors the intended evaluation setup.
        if PROBLEM_MODE == "problem1":
            reject_df = df[df["Str_Label"] == "Reject"]
            n_reject = len(reject_df)
            acc_df = df[df["Str_Label"] == "Accept"].sample(
                n=min(len(df[df["Str_Label"] == "Accept"]), n_reject // 2),
                random_state=42
            )
            rew_df = df[df["Str_Label"] == "Rework"].sample(
                n=min(len(df[df["Str_Label"] == "Rework"]), n_reject - len(acc_df)),
                random_state=42
            )
            df_proc = pd.concat([reject_df, acc_df, rew_df], ignore_index=True)
            df_proc["Str_Label"] = df_proc["Str_Label"].replace({"Accept": "Accept_Rework", "Rework": "Accept_Rework"})

        elif PROBLEM_MODE == "problem2":
            # Keep only Accept and Rework rows.
            df_proc = df[df["Str_Label"].isin(["Accept", "Rework"])].copy()
        else:  # multiclass
            # Keep all three classes as-is.
            df_proc = df.copy()

    # Shuffle rows so class order is not grouped in the saved workbook.
    df_proc = shuffle(df_proc, random_state=42).reset_index(drop=True)

    # Add numeric labels expected by downstream model code.
    df_proc["Labels"] = df_proc["Str_Label"].map(LABEL_MAPS[PROBLEM_MODE])

    # Standardize directory columns based on whether the downstream model expects
    # a single image input or two image inputs.
    dirs = df_proc.apply(lambda row: assign_directories(row, INPUT_MODE), axis=1)
    if INPUT_MODE == "single":
        df_proc["Directories"] = dirs.apply(lambda x: x["Directories"])
        cols = ["Names", "Directories", "Seg_dirs", "Str_Label", "Labels"]
    else:
        df_proc["Directories_1"] = dirs.apply(lambda x: x["Directories_1"])
        df_proc["Directories_2"] = dirs.apply(lambda x: x["Directories_2"])
        cols = ["Names", "Directories_1", "Directories_2", "Seg_dirs", "Str_Label", "Labels"]

    # Keep only the columns used downstream.
    df_proc = df_proc[cols]

    # Save using the existing naming convention so downstream scripts remain compatible.
    out_name = f"AccRejRew_{PROBLEM_MODE}_{INPUT_MODE}_Abby_{split}_v4-3.xlsx"
    df_proc.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"Saved {out_name} - total rows: {len(df_proc)}")
