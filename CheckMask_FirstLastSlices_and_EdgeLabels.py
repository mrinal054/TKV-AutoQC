import os
import nibabel as nib
import numpy as np
import pandas as pd

# =============================================================================
# Script purpose
# =============================================================================
# This script performs two sequential tasks:
#   1. Create per-class image/segmentation log Excel files.
#   2. Analyze each segmentation for:
#        - label presence on the first slice
#        - label presence on the last slice
#        - contact with image boundaries (x/y/z sides)
#
# Expected directory layout for images:
#   image_root_dir/
#       Accept/
#       Reject/
#       Rework/
#
# Segmentations are expected in a flat directory and are matched by filename.
# =============================================================================

# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================
# USER INPUT REQUIRED:
# Root directory containing class subfolders Accept/, Reject/, and Rework/.
image_root_dir = "/path/to/image_root_dir"

# USER INPUT REQUIRED:
# Flat directory containing segmentation .nii.gz files.
seg_root_dir   = "/path/to/segmentation_root_dir"

# USER INPUT REQUIRED:
# Directory where the initial logs and the segmentation-analysis workbooks will
# be written.
output_dir     = "/path/to/output_directory"

# Fixed label encoding used throughout the AutoQC pipeline.
label_map = {"Accept": 0, "Reject": 1, "Rework": 2}
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================
def _write_bold_header_excel(df: pd.DataFrame, path: str, sheet_name: str = "Sheet1"):
    # Write a dataframe to Excel with a bold header row.
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        header_fmt = workbook.add_format({"bold": True})
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)


def _inplane_area_per_voxel_mm2(img: nib.Nifti1Image) -> float:
    # Compute in-plane voxel area from the affine matrix so slice areas can be
    # reported in mm^2.
    voxel_sizes = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    return float(voxel_sizes[0] * voxel_sizes[1])


def _side_touch_info_full(seg_data: np.ndarray):
    # Check whether labels 1 or 2 touch any image boundary and report which
    # sides were touched.
    sides = {
        "x0": seg_data[0, :, :],
        "xN": seg_data[-1, :, :],
        "y0": seg_data[:, 0, :],
        "yN": seg_data[:, -1, :],
        "z0": seg_data[:, :, 0],
        "zN": seg_data[:, :, -1],
    }
    touched = []
    labels_all = []
    for name, sl in sides.items():
        u = np.unique(sl)
        u = u[(u != 0) & np.isin(u, [1, 2])]
        if u.size > 0:
            touched.append(name)
            labels_all.extend(u.tolist())
    labels_all = sorted(set(int(v) for v in labels_all))
    return (len(touched) > 0), touched, labels_all

# =============================================================================
# Stage 1: create initial logs
# =============================================================================
from difflib import get_close_matches


def create_initial_logs():
    # Build a segmentation lookup table from the flat segmentation directory.
    seg_files = [f for f in os.listdir(seg_root_dir) if f.endswith(".nii.gz")]
    seg_dict = {f: os.path.join(seg_root_dir, f) for f in seg_files}

    for label_str, label_int in label_map.items():
        subfolder = os.path.join(image_root_dir, label_str)
        if not os.path.isdir(subfolder):
            print(f"Warning: {subfolder} not found, skipping.")
            continue

        all_files = os.listdir(subfolder)
        image_files = [f for f in all_files if f.endswith("_0000.nii.gz")]

        records = []
        for img_fname in image_files:
            # Remove only the nnUNet channel suffix so the image name can be
            # compared against the segmentation filename.
            img_base = img_fname.replace("_0000", "")

            # Exact filename match first.
            seg_path = seg_dict.get(img_base, "")

            # Fuzzy fallback if the exact segmentation filename is not found.
            if not seg_path:
                matches = get_close_matches(img_base, seg_dict.keys(), n=1, cutoff=0.8)
                if matches:
                    seg_path = seg_dict[matches[0]]
                    print(f"Fuzzy matched '{img_base}' to '{matches[0]}'")

            # Leave Seg_dirs empty if no segmentation could be resolved.
            if not seg_path:
                print(f"No segmentation found for image base: {img_base}")
                seg_path = ""

            records.append({
                "Names": img_fname,
                "Directories": os.path.join(subfolder, img_fname),
                "Labels": label_int,
                "Str_Labels": label_str,
                "Seg_dirs": seg_path
            })

        df = pd.DataFrame(records)
        out_path = os.path.join(output_dir, f"{label_str}_images_and_segs_log.xlsx")
        _write_bold_header_excel(df, out_path)
        print(f"Created initial log for {label_str}: {out_path}")


# =============================================================================
# Stage 2: analyze segmentations
# =============================================================================
def analyze_segmentations():
    # Read each per-class log created above and append slice/edge metrics.
    for label_str, _ in label_map.items():
        log_path = os.path.join(output_dir, f"{label_str}_images_and_segs_log.xlsx")
        if not os.path.isfile(log_path):
            print(f"Log file {log_path} not found, skipping analysis.")
            continue

        df = pd.read_excel(log_path)

        new_cols = [
            "FirstSlice_Label1_Voxels", "FirstSlice_Label1_mm2",
            "FirstSlice_Label2_Voxels", "FirstSlice_Label2_mm2",
            "LastSlice_Label1_Voxels",  "LastSlice_Label1_mm2",
            "LastSlice_Label2_Voxels",  "LastSlice_Label2_mm2",
            "FirstSlice_FoundLabels", "LastSlice_FoundLabels",
            "TouchesSide", "Sides_Touched", "Side_FoundLabels"
        ]

        # Initialize new output columns before processing each row.
        for c in new_cols:
            df[c] = 0 if c.endswith(("_Voxels", "_mm2")) else ""

        for idx, row in df.iterrows():
            # Skip rows that do not have a valid segmentation path.
            seg_path = str(row.get("Seg_dirs", "")).strip()
            if not seg_path or not os.path.isfile(seg_path):
                continue

            # Load segmentation volume and compute in-plane area.
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            area_per_voxel = _inplane_area_per_voxel_mm2(seg_img)

            # Inspect the first and last slices only.
            first_slice = seg_data[:, :, 0]
            last_slice  = seg_data[:, :, -1]

            first_found_labels = []
            last_found_labels  = []

            for label_val in [1, 2]:
                # First-slice metrics for the current label.
                vox_f = int(np.sum(first_slice == label_val))
                mm2_f = vox_f * area_per_voxel
                df.at[idx, f"FirstSlice_Label{label_val}_Voxels"] = vox_f
                df.at[idx, f"FirstSlice_Label{label_val}_mm2"] = mm2_f
                if vox_f > 0:
                    first_found_labels.append(label_val)

                # Last-slice metrics for the current label.
                vox_l = int(np.sum(last_slice == label_val))
                mm2_l = vox_l * area_per_voxel
                df.at[idx, f"LastSlice_Label{label_val}_Voxels"] = vox_l
                df.at[idx, f"LastSlice_Label{label_val}_mm2"] = mm2_l
                if vox_l > 0:
                    last_found_labels.append(label_val)

            if first_found_labels:
                df.at[idx, "FirstSlice_FoundLabels"] = ",".join(str(v) for v in sorted(first_found_labels))
            if last_found_labels:
                df.at[idx, "LastSlice_FoundLabels"] = ",".join(str(v) for v in sorted(last_found_labels))

            # Record whether the segmentation touches any image boundary.
            touches_side, sides_touched, side_labels = _side_touch_info_full(seg_data)
            df.at[idx, "TouchesSide"] = bool(touches_side)
            df.at[idx, "Sides_Touched"] = ";".join(sides_touched) if sides_touched else ""
            if side_labels:
                df.at[idx, "Side_FoundLabels"] = ",".join(str(v) for v in side_labels)

        final_path = os.path.join(output_dir, f"{label_str}_seg_analysis.xlsx")
        _write_bold_header_excel(df, final_path)
        print(f"Saved analysis results for {label_str} to {final_path}")


# =============================================================================
# Run pipeline
# =============================================================================
create_initial_logs()
analyze_segmentations()
