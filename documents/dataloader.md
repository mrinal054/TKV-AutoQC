# Dataloader
---

## [loaderv1](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv1.py?ref_type=heads) — Base implementation
- Initial 3D classification dataloader.
- Handles basic loading, preprocessing, and label handling.

---

## [loaderv2](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv2.py?ref_type=heads) — Zero-padding along Z-axis
- Adds optional **zero-padding along the z-axis** (`n_zSlices`):
  - Pads (or keeps) number of slices to a fixed depth.
  - Uses `zSlices_pad_value` for the padding value.

---

## [loaderv3](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv3.py?ref_type=heads) — Classification types
- Supports three classification modes:
  - `binary`
  - `multiclass`
  - `multilabel`
- Optional **one-hot encoding** for binary and multiclass labels (`onehot=True`).

---

# [loaderv4](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv4.py?ref_type=heads) — Segmentation-aware masking (single-image focus)

- Adds support for **segmentation-driven binary masking**:
  - Reads segmentation labels from `mask_column`.
  - Uses `binary_mask` (list of label values) to create a **binary mask**.
  - Multiplies the image volume by the binary mask to focus on the ROI.
- Introduces `dir_column`:
  - Specifies which dataframe column holds the **image paths**.
  - Example single-image usage:
    - `dir_column="Image_paths"`
    - `mask_column="Seg_dir"`
    - `binary_mask=[1, 2]` → keeps labels 1 and 2 as 1, all others 0.
- Workflow per sample (v4 logic):
  - Load image (`img_path` from `dir_column`).
  - Load segmentation (`mask_path` from `mask_column`) if `binary_mask` is provided.
  - Resample both image and segmentation to common spacing (if `resample` is set).
  - Optionally pad along z-axis (`n_zSlices`).
  - Create binary mask via `Utils3D.binary_mask(label_arr, keep_values=binary_mask)`.
  - Multiply: `img_arr = img_arr * mask` to restrict intensities to masked region.
- This version is especially suited for **single-image + segmentation** workflows.

---

## [loaderv5](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv5.py?ref_type=heads) — Multi-image loading with per-image transforms and masks

- Extends v4 to **multi-image inputs** per sample (e.g., image + mask-ROI image + another modality).
- `dir_column` can now be:
  - A **string** → single-image loader (backwards compatible).
  - A **list of strings** → **multi-image loader**, one column per image path.
    - Example: `dir_column = ["Image_paths", "Mask_paths"]`
- `binary_mask`, `mask_column`, and `transform` can be:
  - Single values (for single-image)
  - **Lists** aligned with `dir_column` length for multi-image:
    - `binary_mask = [None, [1, 2], None]`
    - `mask_column = [None, "Seg_dir", None]`
    - `transform = [None, monai_transform_pipeline, None]`
- For each image channel in `dir_column`:
  - Loads the volume.
  - Optionally:
    - Resamples to desired spacing (`resample`).
    - Pads along z-axis (`n_zSlices`).
    - Applies **intensity clipping** (`clip` or `clip_percentile`).
    - Applies **normalization** (`normalize`).
    - Applies **binary masking** if `binary_mask[i]` and `mask_column[i]` are set.
    - Applies **per-image transforms** (`transform[i]`).
  - Outputs each image as a tensor of shape: `1 x D x H x W`
- **Return format**:
  - **Single-image loader** (string `dir_column` or list of length 1):
    - `img, label, img_name`
    - `img` → tensor `C x D x H x W`
  - **Multi-image loader** (`dir_column` as list with length > 1):
    - `img_list, label, img_names`
    - `img_list` → list of tensors, one per image input
    - `img_names` → list of corresponding file names

---

