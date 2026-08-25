"""
Created on Mon Nov 17 11:22:04 2025

@author: Mrinal Kanti Dhar and Abigail E. Green
"""
#%% Monai transform
import monai.transforms as T
import numpy as np
import torch


class ChannelFirstMONAIWrapper:
    """
    Wrap a MONAI array pipeline so the loader can pass a raw (D,H,W) volume
    while MONAI receives proper channel-first (C,D,H,W) input.

    This fixes MONAI spatial_axis semantics:
        spatial_axis 0 -> D/Z
        spatial_axis 1 -> H/Y
        spatial_axis 2 -> W/X

    The wrapper returns a channel-first tensor and advertises that behavior via
    returns_channel_first=True, so the dataloader does not add a second channel
    dimension.
    """

    returns_channel_first = True
    expects_channel_first = True

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, volume):
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)
        if hasattr(volume, "as_tensor"):
            volume = volume.as_tensor()
        if not torch.is_tensor(volume):
            raise TypeError(f"Expected numpy array or tensor, got {type(volume)}")

        # Input from the loader is usually D,H,W. Convert to C,D,H,W before MONAI.
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        elif volume.ndim == 4:
            # Already channel-first. Leave as-is.
            pass
        else:
            raise ValueError(
                f"Expected a 3D (D,H,W) or 4D (C,D,H,W) volume, got shape {tuple(volume.shape)}"
            )

        out = self.transform(volume.float())
        if hasattr(out, "as_tensor"):
            out = out.as_tensor()
        return out.float()


def monai_pipeline(transform_dict, channel_first: bool = False):
    """
    Build a MONAI Compose pipeline from a transform configuration dictionary.
    
    Each key in `transform_dict` must be the name of a MONAI transform class
    available in `monai.transforms` (for example, "RandFlip", "RandRotate").
    The corresponding value must be a dictionary of keyword arguments that
    will be passed to the transform constructor.
    
    The function dynamically looks up each transform class using `getattr`
    on `monai.transforms`, instantiates it with the provided parameters,
    and combines all created transforms into a `T.Compose` object. The
    transforms are applied in the same order as they appear in the
    `transform_dict` (Python 3.7+ preserves insertion order for dicts).
    
    :param transform_dict: (dict) Mapping from transform name (str) to a dict of
        keyword arguments for that transform. Example:
        {
            "RandFlip": {"spatial_axis": 0, "prob": 0.5},
            "RandRotate": {
                "range_x": 0.25,
                "range_y": 0.25,
                "range_z": 0.25,
                "keep_size": True,
                "mode": "nearest",
                "prob": 0.1},
        }
    :return transform_pipeline: (monai.transforms.Compose) A composed MONAI
        transform that applies the configured transforms sequentially.
    """

    transform_list = []

    for name, params in transform_dict.items():
        transform_class = getattr(T, name) # equivalent to saying e.g. from T import RandFlip
        transform_list.append(transform_class(**params))
    
    transform_list.append(T.ToTensor())

    pipeline = T.Compose(transform_list)
    if channel_first:
        return ChannelFirstMONAIWrapper(pipeline)
    return pipeline


"Example usage"
if __name__ == "__main__":
    import numpy as np
    
    # Configuration dictionary describing which transforms to use and their constructor arguments.
    transform_dict = {
                    "RandFlip": {"spatial_axis": 0, "prob": 0.5},
                    "RandRotate": {"range_x": 0.25, "range_y": 0.25, "range_z": 0.25,
                                   "keep_size": True, "mode": "nearest", "prob": 0.1},
                }

    # Build the MONAI Compose pipeline from the configuration.
    transform_pipeline = monai_pipeline(transform_dict)

    # Example volume
    vol = np.random.rand(128, 128, 64)
    
    # Apply the augmentation pipeline.
    augmented_vol = transform_pipeline(vol)
    
    print("Shape of augmented volume:", augmented_vol.shape)


#%% Numpy and SciPy-based transforms
import numpy as np
from scipy.ndimage import rotate as nd_rotate

class Transform3D:
    @staticmethod
    def flip(volume, axis='random'):
        """
        Flip the volume along a specific or random axis.
        """

        assert axis in [0,1,2] or axis=='random', "Invalid axis. Possible values are - 0,1,2, or random"

        if axis == 'random':
            axis = np.random.choice([0, 1, 2])
            
        flipped = np.flip(volume, axis=axis)
        return flipped.copy() 

    @staticmethod
    def rotate(volume, angle='random', axes='random', interpolation_order=1):
        """
        Rotate the volume by any angle along specific or random axes (plane).
        """
        if angle == 'random':
            angle = np.random.uniform(0, 360) 
        else:
            assert isinstance(angle, (int, float)), "Angle must be a number or 'random'."

        if axes == 'random':
            axes = np.random.choice([(0, 1), (0, 2), (1, 2)])
        else:
            assert axes in [(0, 1), (0, 2), (1, 2)], "Invalid axis. Must be (0, 1), (0, 2), (1, 2), or 'random'."

        return nd_rotate(volume, angle=angle, axes=axes, reshape=False, order=interpolation_order, mode='nearest')

    
    @staticmethod
    def center_crop(volume, crop_size, restore_shape=False, padding=False, interpolation_order=1):
        """
        Center crop the volume to given size.

        :param volume: 3D numpy array.
        :param crop_size: Tuple (H, W, D).
        :param restore_shape: If True, restore to original shape after cropping.
        :param padding: If True and restore_shape is True, restore by zero-padding.
                        If False, restore by resizing using interpolation.
        :param interpolation_order: Used only if restore_shape=True and padding=False.
                                    Order 0 -> Nearest neighbor interpolation
                                    Order 1 -> Trilinear interpolation (for 3D data)
                                    Order 2 -> Quadratic interpolation 
                                    Order 3 -> Cubic interpolation

        :return: Cropped (and optionally restored) volume.
        """
        vol_h, vol_w, vol_d = volume.shape
        crop_h, crop_w, crop_d = crop_size

        assert crop_h <= vol_h and crop_w <= vol_w and crop_d <= vol_d, \
            f"Crop size {crop_size} must be smaller or equal to volume size {volume.shape}."

        start_h = (vol_h - crop_h) // 2
        start_w = (vol_w - crop_w) // 2
        start_d = (vol_d - crop_d) // 2

        cropped = volume[start_h:start_h+crop_h,
                        start_w:start_w+crop_w,
                        start_d:start_d+crop_d].astype(np.float32)

        if restore_shape:
            if padding:
                # Create zero volume and paste the cropped center
                restored = np.zeros_like(volume, dtype=np.float32)
                insert_h = (vol_h - crop_h) // 2
                insert_w = (vol_w - crop_w) // 2
                insert_d = (vol_d - crop_d) // 2

                restored[insert_h:insert_h+crop_h,
                        insert_w:insert_w+crop_w,
                        insert_d:insert_d+crop_d] = cropped
                return restored
            else:
                # Resize back to original using interpolation
                from scipy.ndimage import zoom
                scale_factors = (vol_h / crop_h, vol_w / crop_w, vol_d / crop_d)
                resized = zoom(cropped, scale_factors, order=interpolation_order)
                return resized.astype(np.float32)

        return cropped


class Compose3D:
    def __init__(self, transforms):
        """
        Compose 3D transforms with optional probability.
        
        :param transforms: List of tuples (transform_function, kwargs_dict, p)
            Example:
            [(Transform3D.flip, {'axis': 'random'}, 0.5),
             (Transform3D.rotate, {'angle': 10, 'axis': (1, 2)}, 1.0)]
        """
        self.transforms = transforms

    def __call__(self, volume):
        for transform_func, kwargs, p in self.transforms:
            if np.random.rand() < p:
                volume = transform_func(volume, **kwargs)
        return volume

def pipeline():
    transform_pipeline = Compose3D([
        (Transform3D.flip, {'axis': 'random'}, 0.5),  # 50% chance random flip
        (Transform3D.rotate, {'angle': 20, 'axes': (1, 2)}, 1.0),  # Always rotate 10 degrees along (1, 2)
        (Transform3D.center_crop, {'crop_size': (48, 96, 96), 'restore_shape': True, 'padding': True, 'interpolation_order':1}, 0.0)  # Always center crop
    ])

    return transform_pipeline


# =============================================================================
# "Example usage"
# if __name__ == "__main__":
#     # Compose flexible chain with per-transform control
#     transform_pipeline = Compose3D([
#         (Transform3D.flip, {'axis': 'random'}, 0.5),  # 50% chance random flip
#         (Transform3D.rotate, {'angle': 10, 'axes': (1, 2)}, 1.0),  # Always rotate 10 degrees along (1, 2)
#         (Transform3D.center_crop, {'crop_size': (96, 96, 48), 'restore_shape': False, 'padding': False, 'interpolation_order':1}, 1.0)  # Always center crop
#     ])
# 
#     # transform_pipeline = pipeline()
# 
#     # Example usage in dataset or test
#     vol = np.random.rand(128, 128, 64)
#     augmented_vol = transform_pipeline(vol)
#     print("Shape of augmented volume:", augmented_vol.shape)
# =============================================================================



#%% Multi-input augmentation support
from copy import deepcopy
from collections.abc import Mapping
import torch

# Transforms that change voxel positions and therefore should be synchronized
# across all inputs from the same sample.
SPATIAL_TRANSFORM_NAMES = {
    "RandFlip",
    "RandRotate",
    "RandAffine",
    "RandZoom",
    "Rand3DElastic",
    "RandGridDistortion",
    "RandAxisFlip",
    "RandRotate90",
}

# Transforms that change intensity/appearance and can be restricted to image-like
# inputs. Extend this set when adding new MONAI intensity transforms.
INTENSITY_TRANSFORM_NAMES = {
    "RandScaleIntensity",
    "RandShiftIntensity",
    "RandGaussianNoise",
    "RandGaussianSmooth",
    "RandBiasField",
    "RandAdjustContrast",
    "RandHistogramShift",
    "RandRicianNoise",
    "RandGibbsNoise",
    "RandKSpaceSpikeNoise",
    "RandCoarseDropout",
    "RandCoarseShuffle",
}

# Spatial transforms that interpolate values and therefore need nearest-neighbor
# mode for discrete mask/label inputs.
SPATIAL_TRANSFORMS_WITH_MODE = {
    "RandRotate",
    "RandAffine",
    "RandZoom",
    "Rand3DElastic",
}


def _to_plain_dict(obj):
    """Convert Box-like mappings to plain Python dictionaries recursively."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if isinstance(obj, Mapping):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def split_spatial_and_intensity_transforms(transform_dict):
    """Split a MONAI-style transform config into spatial and intensity groups."""
    transform_dict = _to_plain_dict(transform_dict) or {}
    spatial = {}
    intensity = {}
    other = {}

    for name, params in transform_dict.items():
        if name in SPATIAL_TRANSFORM_NAMES:
            spatial[name] = params or {}
        elif name in INTENSITY_TRANSFORM_NAMES:
            intensity[name] = params or {}
        else:
            other[name] = params or {}

    if other:
        unknown = ", ".join(other.keys())
        raise ValueError(
            "These transforms were not classified as spatial or intensity: "
            f"{unknown}. Add them to SPATIAL_TRANSFORM_NAMES or "
            "INTENSITY_TRANSFORM_NAMES in transforms_v2.py."
        )

    return spatial, intensity


def _normalize_axis_list(spatial_axis):
    """Normalize MONAI spatial_axis into a Python list of axes."""
    if spatial_axis is None:
        return [0, 1, 2]
    if isinstance(spatial_axis, int):
        return [spatial_axis]
    if isinstance(spatial_axis, str):
        if spatial_axis.lower() == "random":
            return [int(torch.randint(low=0, high=3, size=(1,)).item())]
        return [int(spatial_axis)]
    if isinstance(spatial_axis, (list, tuple)):
        return [int(axis) for axis in spatial_axis]
    raise TypeError(f"Unsupported spatial_axis type: {type(spatial_axis)}")


class MultiInputAugmentor:
    """
    Apply one shared spatial augmentation decision to all input tensors from a
    sample, then optionally apply intensity transforms only to selected inputs.

    Expected input/output per item: list of tensors shaped [C, D, H, W].
    MONAI spatial_axis then maps as:
        0 -> D/Z
        1 -> H/Y
        2 -> W/X
    """

    is_multi_input_augmentor = True

    def __init__(
        self,
        transform_dict,
        num_inputs,
        input_types=None,
        intensity_apply_to="images",
        intensity_sync_across_inputs=False,
        mask_interpolation="nearest",
        force_mask_nearest=True,
        left_right_spatial_axis=2,
        label_swap_after_flip=None,
    ):
        self.num_inputs = int(num_inputs)
        self.keys = [f"input_{i}" for i in range(self.num_inputs)]
        self.input_types = self._normalize_input_types(input_types)
        self.intensity_apply_to = intensity_apply_to
        self.intensity_sync_across_inputs = bool(intensity_sync_across_inputs)
        self.mask_interpolation = mask_interpolation
        self.force_mask_nearest = bool(force_mask_nearest)
        self.left_right_spatial_axis = int(left_right_spatial_axis)
        self.label_swap_after_flip = self._normalize_label_swap_map(label_swap_after_flip)

        spatial_dict, intensity_dict = split_spatial_and_intensity_transforms(transform_dict)

        # Handle RandFlip manually so we can guarantee shared flipping and optionally
        # swap 1 <-> 2 labels for a laterality mask after a true left-right flip.
        self.randflip_params = _to_plain_dict(spatial_dict.pop("RandFlip", None))

        self.spatial_pipeline = self._build_dict_pipeline(
            spatial_dict,
            keys=self.keys,
            spatial=True,
        )

        self.intensity_keys = self._resolve_intensity_keys(intensity_apply_to)
        if self.intensity_sync_across_inputs:
            self.intensity_pipeline = self._build_dict_pipeline(
                intensity_dict,
                keys=self.intensity_keys,
                spatial=False,
            )
        else:
            self.intensity_pipeline = self._build_array_pipeline(intensity_dict)

    def _normalize_input_types(self, input_types):
        if input_types is None:
            return ["image"] * self.num_inputs
        input_types = list(input_types)
        if len(input_types) != self.num_inputs:
            raise ValueError(
                f"input_types has length {len(input_types)}, but num_inputs={self.num_inputs}."
            )
        return [str(v).lower() for v in input_types]

    def _normalize_label_swap_map(self, label_swap_after_flip):
        label_swap_after_flip = _to_plain_dict(label_swap_after_flip)
        if not label_swap_after_flip:
            return {}

        normalized = {}
        for input_idx, swap_map in label_swap_after_flip.items():
            idx = int(input_idx)
            normalized[idx] = {float(src): float(dst) for src, dst in swap_map.items()}
        return normalized

    def _is_mask_key(self, key):
        idx = int(key.split("_")[-1])
        return self.input_types[idx] in {"mask", "seg", "label", "segmentation"}

    def _mode_for_key(self, key, default_mode):
        if self.force_mask_nearest and self._is_mask_key(key):
            return self.mask_interpolation
        return default_mode

    def _resolve_intensity_keys(self, intensity_apply_to):
        if intensity_apply_to is None:
            return []

        if isinstance(intensity_apply_to, str):
            mode = intensity_apply_to.lower()
            if mode in {"none", "false", "off"}:
                return []
            if mode == "all":
                return list(self.keys)
            if mode in {"images", "image"}:
                return [k for k in self.keys if not self._is_mask_key(k)]
            raise ValueError(
                "intensity_apply_to must be 'images', 'all', 'none', or a list of input indices."
            )

        indices = [int(v) for v in intensity_apply_to]
        for idx in indices:
            if idx < 0 or idx >= self.num_inputs:
                raise ValueError(f"Invalid intensity input index {idx} for {self.num_inputs} inputs.")
        return [self.keys[idx] for idx in indices]

    def _build_dict_pipeline(self, transform_dict, keys, spatial):
        transform_dict = _to_plain_dict(transform_dict) or {}
        if not transform_dict or not keys:
            return None

        transform_list = []
        for name, params in transform_dict.items():
            params = deepcopy(params or {})
            class_name = f"{name}d"
            transform_class = getattr(T, class_name, None)
            if transform_class is None:
                raise ValueError(
                    f"MONAI dictionary transform {class_name} was not found for {name}."
                )

            params["keys"] = keys

            if spatial and name in SPATIAL_TRANSFORMS_WITH_MODE:
                default_mode = params.get("mode", "bilinear")
                if isinstance(default_mode, (list, tuple)):
                    if len(default_mode) != len(keys):
                        raise ValueError(
                            f"Transform {name} received mode list with length {len(default_mode)}, "
                            f"but there are {len(keys)} keys."
                        )
                    params["mode"] = list(default_mode)
                else:
                    params["mode"] = [self._mode_for_key(k, default_mode) for k in keys]

            transform_list.append(transform_class(**params))

        return T.Compose(transform_list)

    def _build_array_pipeline(self, transform_dict):
        transform_dict = _to_plain_dict(transform_dict) or {}
        if not transform_dict:
            return None

        transform_list = []
        for name, params in transform_dict.items():
            transform_class = getattr(T, name, None)
            if transform_class is None:
                raise ValueError(f"MONAI transform {name} was not found.")
            transform_list.append(transform_class(**(params or {})))

        return T.Compose(transform_list)

    @staticmethod
    def _as_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor-like input, got {type(x)}")
        return x.float()

    def _apply_shared_randflip(self, data):
        if not self.randflip_params:
            return data

        prob = float(self.randflip_params.get("prob", 0.1))
        if torch.rand(()) >= prob:
            return data

        axes = _normalize_axis_list(self.randflip_params.get("spatial_axis", None))
        for axis in axes:
            if axis not in (0, 1, 2):
                raise ValueError(f"RandFlip spatial_axis must be 0, 1, or 2. Got {axis}.")

        # [C,D,H,W], so spatial dims start at tensor dimension 1.
        tensor_dims = [axis + 1 for axis in axes]

        for key in self.keys:
            data[key] = torch.flip(data[key], dims=tensor_dims)

        # Optional laterality correction for raw masks. Only use this if labels are
        # still exact discrete values, for example 1=right kidney and 2=left kidney.
        if self.left_right_spatial_axis in axes and self.label_swap_after_flip:
            for input_idx, swap_map in self.label_swap_after_flip.items():
                key = f"input_{input_idx}"
                if key in data:
                    original = data[key].clone()
                    updated = data[key].clone()
                    for src, dst in swap_map.items():
                        updated[original == src] = dst
                    data[key] = updated

        return data

    def __call__(self, tensors):
        if len(tensors) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} tensors, got {len(tensors)}.")

        data = {key: self._as_tensor(tensor) for key, tensor in zip(self.keys, tensors)}

        # Shared spatial transforms first.
        data = self._apply_shared_randflip(data)
        if self.spatial_pipeline is not None:
            data = self.spatial_pipeline(data)

        # Optional intensity transforms after spatial transforms.
        if self.intensity_pipeline is not None and self.intensity_keys:
            if self.intensity_sync_across_inputs:
                subset = {key: data[key] for key in self.intensity_keys}
                subset = self.intensity_pipeline(subset)
                for key in self.intensity_keys:
                    data[key] = subset[key]
            else:
                for key in self.intensity_keys:
                    data[key] = self.intensity_pipeline(data[key])

        out = []
        for key in self.keys:
            tensor = data[key]
            if hasattr(tensor, "as_tensor"):
                tensor = tensor.as_tensor()

            # Keep mask-like branches label-preserving after shared spatial transforms.
            # This is safe for binary masks and regular 0/1/2 masks when the mask branch
            # has been preprocessed with nearest-neighbor interpolation.
            if self._is_mask_key(key):
                tensor = torch.round(tensor)

            out.append(tensor.float())
        return out


def make_multi_input_augmentor(
    transform_dict,
    num_inputs,
    input_types=None,
    intensity_apply_to="images",
    intensity_sync_across_inputs=False,
    mask_interpolation="nearest",
    force_mask_nearest=True,
    left_right_spatial_axis=2,
    label_swap_after_flip=None,
):
    """Factory used by trainer_multiImageNet_v1_1.py."""
    return MultiInputAugmentor(
        transform_dict=transform_dict,
        num_inputs=num_inputs,
        input_types=input_types,
        intensity_apply_to=intensity_apply_to,
        intensity_sync_across_inputs=intensity_sync_across_inputs,
        mask_interpolation=mask_interpolation,
        force_mask_nearest=force_mask_nearest,
        left_right_spatial_axis=left_right_spatial_axis,
        label_swap_after_flip=label_swap_after_flip,
    )
