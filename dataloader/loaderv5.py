"""
Mask-aware, user-driven v5 dataloader for TKV AutoQC.

Key additions over the original loaderv5.py:
  - input_types: per-input type, e.g. ["image", "mask", "image"]
  - mask_input_modes: per-input mask representation, e.g. [None, "label", None]
      * "label"/"preserve": preserve discrete labels such as 0/1/2
      * "binary": convert selected mask labels to 0/1
  - mask_keep_values: per-input keep values for binary mask inputs, e.g. [None, [1, 2], None]
  - preserve_mask_labels: if True, mask inputs use nearest-neighbor resampling/resizing,
    skip clipping/normalization, and are rounded back to integer-like values.
  - shared multi-input augmentor support from the augmentation-fix path.
"""

import sys
sys.path.append("/research/m324371/Project/Digital_Twin/Classification/utils/")

from utils import Utils3D
from typing import Union, List
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk


_MASK_TYPE_NAMES = {"mask", "seg", "label", "segmentation"}
_LABEL_MASK_MODES = {"label", "labels", "preserve", "regular", "categorical"}
_BINARY_MASK_MODES = {"binary", "binarize", "binarized"}


class ClsDataset(Dataset):
    def __init__(self,
                 dataframe,
                 dir_column: Union[List[str], str] = "Directories",
                 label_column: Union[List[str], str] = None,
                 classification_type: str = "binary",
                 binary_mask: Union[List, List[List]] = None,
                 mask_column: Union[str, List[str]] = None,
                 onehot: bool = False,
                 resample: tuple = (1.0, 1.0, 1.0),
                 n_zSlices: int = None,
                 zSlices_pad_value: int = 0,
                 clip: tuple = (-1000, 400),
                 clip_percentile: tuple = None,
                 normalize: bool = True,
                 resize: tuple = (64, 128, 128),
                 resize_method: str = "interpolation",
                 resize_pad_value: int = -1,
                 transform=None,
                 input_types=None,
                 preserve_mask_labels: bool = True,
                 mask_input_modes=None,
                 mask_keep_values=None,
                 verbose: bool = False):
        """
        Unified dataset for binary, multiclass, and multilabel classification.

        Existing behavior is preserved when input_types is None. In that case, every
        input is treated as an image-like volume, matching the original loader.

        New mask-aware fields:
            input_types: list like ["image", "mask", "image"]. Only entries marked
                as mask/seg/label/segmentation are treated as masks.
            preserve_mask_labels: if True, mask inputs skip image-intensity clipping
                and normalization and use nearest-neighbor resampling/resizing.
            mask_input_modes: list like [None, "label", None] or [None, "binary", None].
                "label" preserves labels such as 0/1/2; "binary" converts selected
                values to 0/1.
            mask_keep_values: list like [None, [1, 2], None], used when a mask input
                has mode "binary". If omitted, binary_mask for that input is used as
                a fallback keep-value list.
        """
        self.df = dataframe
        self.dir_column = dir_column
        self.label_column = label_column
        self.classification_type = classification_type
        self.binary_mask = binary_mask
        self.mask_column = mask_column
        self.onehot = onehot
        self.resample = resample
        self.n_zSlices = n_zSlices
        self.zSlices_pad_value = zSlices_pad_value
        self.clip = clip
        self.clip_percentile = clip_percentile
        self.normalize = normalize
        self.resize = resize
        self.resize_method = resize_method
        self.resize_pad_value = resize_pad_value
        self.transform = transform
        self.input_types = input_types
        self.preserve_mask_labels = bool(preserve_mask_labels)
        self.mask_input_modes = mask_input_modes
        self.mask_keep_values = mask_keep_values
        self.verbose = verbose

        if classification_type not in ["binary", "multiclass", "multilabel"]:
            raise ValueError("classification_type must be 'binary', 'multiclass', or 'multilabel'")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _is_multi_input_augmentor(transform):
        """Return True for sample-level multi-input augmentation objects."""
        return getattr(transform, "is_multi_input_augmentor", False)

    def _apply_multi_input_augmentor_if_needed(self, img_arr):
        """Apply a sample-level augmentor to a single-image case if requested."""
        if self._is_multi_input_augmentor(self.transform):
            return self.transform([img_arr])[0]
        return img_arr

    @staticmethod
    def _is_mask_type(input_type) -> bool:
        return str(input_type).lower() in _MASK_TYPE_NAMES

    @staticmethod
    def _normalize_mask_mode(mask_mode):
        if mask_mode is None:
            return "label"
        mode = str(mask_mode).lower()
        if mode in _LABEL_MASK_MODES:
            return "label"
        if mode in _BINARY_MASK_MODES:
            return "binary"
        raise ValueError(
            f"Unsupported mask_input_mode={mask_mode!r}. Use 'label'/'preserve' or 'binary'."
        )

    @staticmethod
    def _as_list_for_inputs(value, n_inputs, default=None, name="value"):
        """Expand scalar/None/list configuration to length n_inputs."""
        if value is None:
            return [default] * n_inputs
        if isinstance(value, list):
            if len(value) != n_inputs:
                raise ValueError(f"{name} has length {len(value)}, expected {n_inputs}.")
            return value
        return [value] * n_inputs

    @staticmethod
    def _pad_z(volume, target_depth, pad_value):
        current_depth = volume.shape[0]
        if current_depth >= target_depth:
            return volume
        pad_total = target_depth - current_depth
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(
            volume,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

    def _resample_if_needed(self, img_obj, is_mask_input=False):
        """Resample SimpleITK image and return object, array, and spacing used."""
        resample_sitk_order = None
        if self.resample:
            # User gave resample as (D, H, W). Reverse it for SimpleITK (W, H, D).
            resample_sitk_order = list(reversed(self.resample))

            if resample_sitk_order[0] is None:
                resample_sitk_order[0] = img_obj.GetSpacing()[0]
            if resample_sitk_order[1] is None:
                resample_sitk_order[1] = img_obj.GetSpacing()[1]
            if resample_sitk_order[2] is None:
                resample_sitk_order[2] = img_obj.GetSpacing()[2]

            interpolator = sitk.sitkNearestNeighbor if is_mask_input else sitk.sitkLinear
            img_obj, img_arr = Utils3D.resample(
                img_obj,
                new_spacing=resample_sitk_order,
                interpolator=interpolator,
            )
        else:
            img_arr = sitk.GetArrayFromImage(img_obj)

        return img_obj, img_arr, resample_sitk_order

    def _resize_if_needed(self, img_arr, metadata, is_mask_input=False):
        if not self.resize:
            return img_arr

        if self.resize_method == "interpolation":
            resize_order = 0 if is_mask_input else 1
            img_arr, _ = Utils3D.resize(
                img_arr,
                desired_width=self.resize[2],
                desired_height=self.resize[1],
                desired_depth=self.resize[0],
                order=resize_order,
                original_spacing=metadata["Spacing"],
            )
        elif self.resize_method == "center_crop":
            pad_value = 0 if is_mask_input else self.resize_pad_value
            img_arr = Utils3D.resize_with_center_crop(
                img_arr,
                desired_width=self.resize[2],
                desired_height=self.resize[1],
                desired_depth=self.resize[0],
                pad_value=pad_value,
            )
        else:
            raise ValueError("Unsupported keyword for resize. Supported are: interpolation and center_crop.")

        return img_arr

    def _read_primary_volume(self, img_path):
        img_obj, img_arr, metadata = Utils3D.read_nifti(img_path)

        if self.verbose:
            print(
                f"Original image shape in (D,H,W): {img_arr.shape}\n"
                f"Original spacing in (Sx,Sy,Sz) or (W,H,D): {metadata['Spacing']}"
            )

        # Convert ChxDxHxW to DxHxW in rare cases.
        if img_arr.ndim == 4:
            img_arr = img_arr[0]
            if self.verbose:
                print(
                    f"Original image is 4D. Taking first channel; new shape is {img_arr.shape}."
                )

            size = list(img_obj.GetSize())  # [X, Y, Z, C]
            size[-1] = 0
            index = [0] * len(size)
            img_obj = sitk.Extract(img_obj, size=size, index=index)

        return img_obj, img_arr, metadata

    def _load_single_image(
        self,
        img_path,
        binary_mask=None,
        mask_path=None,
        transform=None,
        input_type="image",
        mask_input_mode=None,
        mask_keep_values=None,
    ):
        """Load/preprocess one input and return (tensor CxDxHxW, name_str)."""
        img_name = os.path.basename(img_path)
        is_mask_input = self.preserve_mask_labels and self._is_mask_type(input_type)
        mask_mode = self._normalize_mask_mode(mask_input_mode) if is_mask_input else None

        img_obj, img_arr, metadata = self._read_primary_volume(img_path)

        # Resample primary volume. Masks use nearest-neighbor; images use linear.
        img_obj, img_arr, resample_sitk_order = self._resample_if_needed(
            img_obj,
            is_mask_input=is_mask_input,
        )
        if self.verbose:
            print("Resampled image shape in (D,H,W):", img_arr.shape)

        # Z-padding. Masks always pad with background 0.
        if self.n_zSlices:
            before = img_arr.shape[0]
            pad_value = 0 if is_mask_input else self.zSlices_pad_value
            img_arr = self._pad_z(img_arr, self.n_zSlices, pad_value)
            if self.verbose and img_arr.shape[0] != before:
                print(f"Padded along Z-axis from {before} to {img_arr.shape[0]} slices")

        # For mask inputs, optionally convert selected labels to binary.
        # If mask_keep_values is omitted, binary_mask is accepted as a fallback.
        if is_mask_input:
            if mask_mode == "binary":
                keep_values = mask_keep_values if mask_keep_values is not None else binary_mask
                if keep_values is None:
                    raise ValueError(
                        f"Mask input {img_name} requested mask_input_mode='binary' but no "
                        "mask_keep_values or binary_mask keep-values were provided."
                    )
                img_arr = Utils3D.binary_mask(img_arr, keep_values=keep_values).astype(np.int16)
            else:
                # Preserve labels such as 0/1/2.
                img_arr = np.rint(img_arr).astype(np.int16)
        else:
            # Image-only intensity preprocessing.
            if self.clip:
                img_arr = Utils3D.clip_intensity(img_arr, self.clip)
            elif self.clip_percentile:
                lower_p = np.percentile(img_arr, self.clip_percentile[0])
                upper_p = np.percentile(img_arr, self.clip_percentile[1])
                img_arr = np.clip(img_arr, lower_p, upper_p)
                if self.verbose:
                    print("Percentile values:", lower_p, upper_p)

            if self.normalize:
                img_arr = Utils3D.normalize(img_arr)

        # Apply ROI mask to image-like branches only. For mask inputs, binary_mask is
        # treated as keep-values only when mask_input_mode='binary'; it is not used
        # as an ROI multiplier.
        if (not is_mask_input) and binary_mask:
            assert isinstance(binary_mask, (list, tuple)), "binary_mask should be a list or tuple."
            if mask_path is None:
                raise ValueError(f"binary_mask was provided for {img_name}, but mask_path is None.")

            label_obj, label_arr, label_meta = Utils3D.read_nifti(mask_path)

            # Resample segmentation with nearest-neighbor to match image spacing.
            if self.resample:
                label_obj, label_arr = Utils3D.resample(
                    label_obj,
                    new_spacing=resample_sitk_order,
                    interpolator=sitk.sitkNearestNeighbor,
                )

            # Pad label along Z-axis to match image.
            if self.n_zSlices:
                before = label_arr.shape[0]
                label_arr = self._pad_z(label_arr, self.n_zSlices, pad_value=0)
                if self.verbose and label_arr.shape[0] != before:
                    print(f"Padded label along Z-axis from {before} to {label_arr.shape[0]} slices")

            label_arr = np.rint(label_arr).astype(np.int16)
            if label_arr.shape != img_arr.shape:
                raise ValueError(
                    f"Segmentation and image shape mismatch for {img_name}: "
                    f"{label_arr.shape} vs {img_arr.shape}"
                )

            roi_mask = Utils3D.binary_mask(label_arr, keep_values=binary_mask).astype(img_arr.dtype)
            img_arr = img_arr * roi_mask

        # Resize after intensity preprocessing/ROI masking. Mask inputs use nearest.
        img_arr = self._resize_if_needed(img_arr, metadata, is_mask_input=is_mask_input)
        if self.verbose and self.resize:
            print("Resized image shape in (D,H,W):", img_arr.shape)

        # Final safety for mask inputs: restore exact integer-like labels after
        # nearest-neighbor resize/crop.
        if is_mask_input:
            img_arr = np.rint(img_arr).astype(np.float32)
        else:
            img_arr = img_arr.astype(np.float32, copy=False)

        # Per-input transforms. Legacy channel-first wrappers may return C,D,H,W.
        transform_returns_channel_first = False
        if transform is not None:
            transform_returns_channel_first = bool(getattr(transform, "returns_channel_first", False))
            img_arr = transform(img_arr)

        if isinstance(img_arr, np.ndarray):
            img_arr = torch.from_numpy(img_arr)
        if hasattr(img_arr, "as_tensor"):
            img_arr = img_arr.as_tensor()

        # Standardize to C x D x H x W.
        if img_arr.ndim == 3:
            img_arr = img_arr.unsqueeze(0)
        elif img_arr.ndim == 4:
            if not transform_returns_channel_first and self.verbose:
                print(
                    "[WARN] Received a 4D tensor from transform without "
                    "returns_channel_first=True; assuming C,D,H,W."
                )
        else:
            raise ValueError(
                f"Expected 3D or 4D image tensor after preprocessing, got shape {tuple(img_arr.shape)}"
            )

        return img_arr.float(), img_name

    def _get_label(self, row):
        if self.classification_type == "multilabel":
            label = row[self.label_column].values.astype(np.float32)
            return torch.from_numpy(label)

        if isinstance(self.label_column, list):
            label = row[self.label_column[0]]
        else:
            label = row[self.label_column]

        if self.classification_type == "binary":
            label = torch.tensor(label).float()
            if self.onehot:
                label = torch.tensor([1.0, 0.0]) if label.item() == 0 else torch.tensor([0.0, 1.0])
        elif self.classification_type == "multiclass":
            label = torch.tensor(label).long()
            if self.onehot:
                num_classes = len(set(self.df[self.label_column]))
                onehot_vec = torch.zeros(num_classes)
                onehot_vec[label] = 1.0
                label = onehot_vec
        else:
            raise ValueError(f"Unsupported classification_type={self.classification_type}")

        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self._get_label(row)

        # Single-image path.
        if isinstance(self.dir_column, str) or (isinstance(self.dir_column, list) and len(self.dir_column) == 1):
            if isinstance(self.dir_column, str):
                dir_col = self.dir_column
                binary_mask = self.binary_mask if isinstance(self.binary_mask, (list, tuple)) else None
                mask_col = self.mask_column if isinstance(self.mask_column, str) else None
                input_type = self.input_types[0] if isinstance(self.input_types, list) else (self.input_types or "image")
                mask_input_mode = self.mask_input_modes[0] if isinstance(self.mask_input_modes, list) else self.mask_input_modes
                mask_keep_values = self.mask_keep_values[0] if isinstance(self.mask_keep_values, list) else self.mask_keep_values
            else:
                dir_col = self.dir_column[0]
                binary_mask = self.binary_mask[0] if isinstance(self.binary_mask, list) else self.binary_mask
                mask_col = self.mask_column[0] if isinstance(self.mask_column, list) else self.mask_column
                input_type = self.input_types[0] if isinstance(self.input_types, list) else (self.input_types or "image")
                mask_input_mode = self.mask_input_modes[0] if isinstance(self.mask_input_modes, list) else self.mask_input_modes
                mask_keep_values = self.mask_keep_values[0] if isinstance(self.mask_keep_values, list) else self.mask_keep_values

            img_path = row[dir_col]
            mask_path = row[mask_col] if isinstance(mask_col, str) else None
            use_shared_augmentor = self._is_multi_input_augmentor(self.transform)

            img_arr, img_name = self._load_single_image(
                img_path=img_path,
                binary_mask=binary_mask,
                mask_path=mask_path,
                transform=None if use_shared_augmentor else self.transform,
                input_type=input_type,
                mask_input_mode=mask_input_mode,
                mask_keep_values=mask_keep_values,
            )
            img_arr = self._apply_multi_input_augmentor_if_needed(img_arr)
            return img_arr, label, img_name

        # Multi-image path.
        n_inputs = len(self.dir_column)
        binary_mask = self._as_list_for_inputs(self.binary_mask, n_inputs, default=None, name="binary_mask")
        mask_column = self._as_list_for_inputs(self.mask_column, n_inputs, default=None, name="mask_column")
        input_types = self._as_list_for_inputs(self.input_types, n_inputs, default="image", name="input_types")
        mask_input_modes = self._as_list_for_inputs(self.mask_input_modes, n_inputs, default=None, name="mask_input_modes")
        mask_keep_values = self._as_list_for_inputs(self.mask_keep_values, n_inputs, default=None, name="mask_keep_values")

        use_shared_augmentor = self._is_multi_input_augmentor(self.transform)
        if use_shared_augmentor:
            transforms = [None] * n_inputs
        elif self.transform is None or isinstance(self.transform, bool):
            transforms = [self.transform] * n_inputs
        else:
            transforms = self.transform

        if len(transforms) != n_inputs:
            raise ValueError(
                f"Length of transform ({len(transforms)}) must match dir_column ({n_inputs})."
            )

        img_arrs, img_names = [], []
        for i, (dir_col, bin_mask, mask_col, transform, input_type, mask_mode, keep_values) in enumerate(
            zip(self.dir_column, binary_mask, mask_column, transforms, input_types, mask_input_modes, mask_keep_values)
        ):
            img_path = row[dir_col]
            mask_path = row[mask_col] if isinstance(mask_col, str) else None

            img_arr, img_name = self._load_single_image(
                img_path=img_path,
                binary_mask=bin_mask,
                mask_path=mask_path,
                transform=None if use_shared_augmentor else transform,
                input_type=input_type,
                mask_input_mode=mask_mode,
                mask_keep_values=keep_values,
            )
            img_arrs.append(img_arr)
            img_names.append(img_name)

        if use_shared_augmentor:
            img_arrs = self.transform(img_arrs)

        return img_arrs, label, img_names
