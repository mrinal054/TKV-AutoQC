"""
Created on Mon Nov 17 11:22:04 2025

@author: m324371
"""
#%% Monai transform
import monai.transforms as T

def monai_pipeline(transform_dict):
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
    
    return T.Compose(transform_list)


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

