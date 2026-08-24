import os

import pandas as pd
from sklearn.utils import shuffle

# =============================================================================
# Script purpose
# =============================================================================
# This script converts the patient-disjoint train/val/test log Excel files from
# the standardized experimental set into model-input Excel files for downstream
# AutoQC training and evaluation.
#
# Expected input columns in each source workbook:
#   - Names
#   - Str_Label
#   - Seg_dirs or Seg_Dirs
#   - Directories or Directories_1 / Directories_2 or Directories1 / Directories2
#
# The script supports three label-grouping modes:
#   - "problem1": Accept/Rework vs Reject
#   - "problem2": Accept vs Rework
#   - "multiclass": Accept vs Reject vs Rework
#
# This script does not create or modify split assignments. It verifies that the
# upstream source logs are patient-disjoint, then preserves each row's split.
# =============================================================================

# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================
PROBLEM_MODE = "multiclass"  # Choose from: "problem1", "problem2", "multiclass"
INPUT_MODE = "multi"         # Choose from: "single", "multi"
USE_FULL_TRAIN = False       # Only affects training split selection logic for some problem modes

# Preferred: specify a deidentified patient/group column present in all three
# source logs. If None, the first underscore-delimited token in Names is used.
PATIENT_ID_COLUMN = None
PATIENT_ID_NAME_SEPARATOR = "_"

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
    "val": os.path.join(BASE_DIR, "AccRejRew_StandardExpSet_val_log.xlsx"),
    "test": os.path.join(BASE_DIR, "AccRejRew_StandardExpSet_test_log.xlsx"),
}

# Numeric label mappings used by the downstream training code.
LABEL_MAPS = {
    "problem1": {"Reject": 0, "Accept_Rework": 1},
    "problem2": {"Accept": 1, "Rework": 0},
    "multiclass": {"Accept": 0, "Reject": 1, "Rework": 2},
}

# =============================================================================
# Patient-disjoint verification
# =============================================================================
def get_patient_groups(df, source_name):
    """Derive patient groups without changing the saved output schema."""
    if PATIENT_ID_COLUMN is not None:
        if PATIENT_ID_COLUMN not in df.columns:
            raise KeyError(
                f"{source_name} does not contain PATIENT_ID_COLUMN="
                f"{PATIENT_ID_COLUMN!r}."
            )
        groups = df[PATIENT_ID_COLUMN]
    else:
        if "Names" not in df.columns:
            raise KeyError(
                f"{source_name} must contain 'Names' when PATIENT_ID_COLUMN is None."
            )
        groups = (
            df["Names"]
            .astype(str)
            .str.split(PATIENT_ID_NAME_SEPARATOR, n=1)
            .str[0]
        )

    groups = groups.astype(str).str.strip()
    invalid = groups.eq("") | groups.str.lower().eq("nan")
    if invalid.any():
        bad_rows = invalid[invalid].index[:10].tolist()
        raise ValueError(
            f"Unable to determine patient group for rows {bad_rows} in {source_name}."
        )

    return groups


def verify_patient_disjoint_input_logs(input_files):
    """Fail if the source train/val/test logs contain patient overlap."""
    groups_by_split = {}

    for split, input_path in input_files.items():
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Missing {split} input log: {input_path}")

        df_check = pd.read_excel(input_path)
        groups_by_split[split] = set(get_patient_groups(df_check, input_path))

    overlaps = {
        "train/val": groups_by_split["train"] & groups_by_split["val"],
        "train/test": groups_by_split["train"] & groups_by_split["test"],
        "val/test": groups_by_split["val"] & groups_by_split["test"],
    }
    overlaps = {name: values for name, values in overlaps.items() if values}

    if overlaps:
        summary = {name: len(values) for name, values in overlaps.items()}
        raise RuntimeError(
            "Patient leakage detected in source split logs: "
            f"{summary}"
        )

    print("Patient-disjoint source log check passed:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(groups_by_split[split])} patients")


# =============================================================================
# Directory-column compatibility helpers
# =============================================================================
def _first_available_value(row, candidates, default=""):
    """Return the first non-null value from compatible column names."""
    for column in candidates:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return default


def assign_directories(row, input_mode):
    # Standardize directory columns so downstream code sees the expected schema.
    # The compatibility aliases preserve the existing curator output names while
    # also accepting newer underscore-based names.
    directory_1 = _first_available_value(
        row,
        ["Directories_1", "Directories1", "Directories"],
    )
    directory_2 = _first_available_value(
        row,
        ["Directories_2", "Directories2", "Directories"],
        default=directory_1,
    )

    if input_mode == "single":
        return {"Directories": directory_1}

    return {
        "Directories_1": directory_1,
        "Directories_2": directory_2,
    }


# =============================================================================
# Verify fixed split assignments before any task-specific filtering
# =============================================================================
verify_patient_disjoint_input_logs(INPUT_FILES)

# =============================================================================
# Process each split
# =============================================================================
for split in ["train", "val", "test"]:
    print(f"\n--- Processing {split} ({PROBLEM_MODE}) ---")

    # Read the split workbook and normalize string labels for consistent matching.
    df = pd.read_excel(INPUT_FILES[split])
    df["Str_Label"] = df["Str_Label"].astype(str).str.title()

    # Preserve compatibility with the curator's existing Seg_Dirs output name.
    if "Seg_dirs" not in df.columns and "Seg_Dirs" in df.columns:
        df["Seg_dirs"] = df["Seg_Dirs"]

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
                acc_sample = acc_df.sample(
                    n=min(n_reject // 2, len(acc_df)),
                    random_state=42,
                )
                rew_sample = rew_df.sample(
                    n=min(n_reject - len(acc_sample), len(rew_df)),
                    random_state=42,
                )
                df_proc = pd.concat(
                    [reject_df, acc_sample, rew_sample],
                    ignore_index=True,
                )

            # Merge Accept and Rework into a single positive class.
            df_proc["Str_Label"] = df_proc["Str_Label"].replace(
                {"Accept": "Accept_Rework", "Rework": "Accept_Rework"}
            )

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
                random_state=42,
            )
            rew_df = df[df["Str_Label"] == "Rework"].sample(
                n=min(len(df[df["Str_Label"] == "Rework"]), n_reject - len(acc_df)),
                random_state=42,
            )
            df_proc = pd.concat([reject_df, acc_df, rew_df], ignore_index=True)
            df_proc["Str_Label"] = df_proc["Str_Label"].replace(
                {"Accept": "Accept_Rework", "Rework": "Accept_Rework"}
            )

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
        cols = [
            "Names",
            "Directories_1",
            "Directories_2",
            "Seg_dirs",
            "Str_Label",
            "Labels",
        ]

    # Keep only the columns used downstream.
    df_proc = df_proc[cols]

    # Preserve the existing output filename convention so downstream scripts and
    # experiment configs remain compatible.
    out_name = f"AccRejRew_{PROBLEM_MODE}_{INPUT_MODE}_Abby_{split}_v4-3.xlsx"
    df_proc.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"Saved {out_name} - total rows: {len(df_proc)}")
