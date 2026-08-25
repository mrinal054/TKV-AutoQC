# Dataloader

## [loaderv5](../dataloader/loaderv5.py) — Multi-image loading with per-image transforms and masks

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

