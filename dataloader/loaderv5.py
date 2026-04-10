"""
Mrinal @ 6 June 2025

Last modified @ 12 November 2025

v1: Base implementation
v2: zero-padding along z-axis after resampling added (n_zSlices)
v3: Supports binary, multi-class, and multi-label classification
v4: Reads segmentation labels, creates binary mask, multiplies image with binary mask
    Param dir_column added.
v5: Supports multi-image loading with separate transforms and binary masking for each individual image.
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

class ClsDataset(Dataset):
    def __init__(self, 
                 dataframe,
                 dir_column: Union[List[str], str] = "Directories",
                 label_column: Union[List[str], str] = None,
                 classification_type:str = "binary",  # 'binary', 'multiclass', or 'multilabel'
                 binary_mask: Union[List, List[List]] = None,
                 mask_column: Union[str, List[str]] = None,
                 onehot:bool = False,                  # For binary/multiclass classification
                 resample:tuple = (1.0, 1.0, 1.0),
                 n_zSlices:int = None,
                 zSlices_pad_value:int = 0,
                 clip:tuple = (-1000, 400),
                 clip_percentile:tuple = None,
                 normalize:bool = True,
                 resize:tuple = (64, 128, 128),
                 resize_method:str = "interpolation", # supports interpolation and center_crop
                 resize_pad_value:int = -1, # used in center_crop 
                 transform = None,
                 verbose:bool = False):
        """
        Unified dataset for binary, multiclass, and multilabel classification.

        :param dataframe: DataFrame with at least 'Directories' and 'Labels' (or label columns for multilabel)
        :param dir_column: Name of the column in the dataframe that lists image directories (full path).
                           For multi-image loader, dir_column is a list of column names .
                            Example for single-image loader: Image_paths
                            Example for multi-image loader: [Image_paths, Mask_paths]
        :param label_column: List (for multi-label) or str (for binary/multi-class) of column names that stores labels.
        :param classification_type: Type of classification: 'binary', 'multiclass', or 'multilabel'
        :param binary_mask: List or List of lists. For multi-image loader, length of dir_column and binary_mask
                            have to be the same. 
                            It creates binary mask using the list. e.g. for [1,2], 
                            it will convert pixels 1 & 2 to 1, and remaining pixels to 0.
                            If not None, later image will be multiplied by the mask.
                            Example for single-image loader: [1,2]
                            Example for multi-image loader: [None, [1,2], None] or None
        :param mask_column: (list of str, optional) Path to the segmentation mask, used when binary_mask is provided.
                            Example for single-image loader: Seg_dir
                            Example for multi-image loader: [None, Seg_dir, None] or None
        :param onehot: Whether to one-hot encode labels (for binary or multiclass)
        :param resample: Tuple (D, H, W) for spacing; if None, skips resampling
        :param n_zSlices: If specified, pads to a fixed number of slices along Z
        :param zSlices_pad_value: Value used for padding along z-axis
        :param clip: Tuple to clip image intensities (min, max). 
        :param clip_percentile: Tuple to clip image intensities based on percentile value. Only works when clip is set to None. 
        :param normalize: Normalize intensity values
        :param resize: Output shape (D, H, W)
        :param resize_method: the way resizing will be performed. Currently, supports interpolation and center_crop
        :param resize_pad_value: Value used for padding in center cropping
        :param transform: Optional transforms
        :param verbose: Whether to print debugging info
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
        self.verbose = verbose

        if classification_type not in ["binary", "multiclass", "multilabel"]:
            raise ValueError("classification_type must be 'binary', 'multiclass', or 'multilabel'")

    def __len__(self):
        return len(self.df)
    
    # Helper: load and preprocess ONE image
    def _load_single_image(self, img_path, binary_mask=None, mask_path=None, transform=None):
        """Return (tensor CxDxHxW, name_str)."""
        img_name = os.path.basename(img_path)
        
        # Read NIfTI image
        img_obj, img_arr, metadata = Utils3D.read_nifti(img_path)
        if self.verbose: print(f"Original image shape in (D,H,W): {img_arr.shape} \n Original spacing in (Sx,Sy,Sz) or (W,H,D): {metadata['Spacing']}")

        # Convert ChxDxHxW to DxHxW (rare cases)
        if img_arr.ndim == 4: 
            img_arr = img_arr[0]
            if self.verbose: print(f"Original image is {img_arr.ndim} with "
                                   f"{img_arr.shape[0]} channels. Taking only first channel")

            # Need to update the object file as well. Necessay for resampling
            size = list(img_obj.GetSize())  # [X, Y, Z, C]
            size[-1] = 0  # Collapse last dimension (channel)
            index = [0] * len(size)

            img_obj = sitk.Extract(img_obj, size=size, index=index)
            
        # Resample 
        if self.resample:
            # User gave resample as (D, H, W). Reverse it for SimpleITK (W, H, D)
            resample_sitk_order = list(reversed(self.resample))

            # If sampling is None, then set to original spacing
            if resample_sitk_order[0] is None: resample_sitk_order[0] = img_obj.GetSpacing()[0]
            if resample_sitk_order[1] is None: resample_sitk_order[1] = img_obj.GetSpacing()[1]
            if resample_sitk_order[2] is None: resample_sitk_order[2] = img_obj.GetSpacing()[2]

            img_obj, img_arr = Utils3D.resample(img_obj, new_spacing=resample_sitk_order)
            if self.verbose: print("Resampled image shape in (D,H,W):", img_arr.shape)        
            
        # Zero-padding along Z-axis if n_zSlices is specified
        if self.n_zSlices:
            current_depth = img_arr.shape[0]
            if current_depth < self.n_zSlices:
                pad_total = self.n_zSlices - current_depth
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                img_arr = np.pad(img_arr, 
                                ((pad_before, pad_after), (0, 0), (0, 0)), 
                                mode='constant', constant_values=self.zSlices_pad_value)
                if self.verbose:
                    print(f"Padded along Z-axis from {current_depth} to {img_arr.shape[0]} slices")            

        # Clip intensity
        if self.clip:
            img_arr = Utils3D.clip_intensity(img_arr, self.clip)
        elif self.clip_percentile:
            lower_p = np.percentile(img_arr, self.clip_percentile[0])
            upper_p = np.percentile(img_arr, self.clip_percentile[1])
            img_arr = np.clip(img_arr, lower_p, upper_p)

            if self.verbose:
                print("Percentile values:", lower_p, upper_p)        
                
        # Normalize intensity
        if self.normalize:
            img_arr = Utils3D.normalize(img_arr)

        # Apply binary mask (after resampling)
        if binary_mask:
            assert isinstance(binary_mask, (list, tuple)), "binary_mask should be a list or tuple."

            label_obj, label_arr, label_meta = Utils3D.read_nifti(mask_path)

            # Resample segmentation to match image spacing
            if self.resample:
                label_obj, label_arr = Utils3D.resample(label_obj, new_spacing=resample_sitk_order)

            # Pad label along Z-axis to match image, if n_zSlices is specified
            if self.n_zSlices:
                label_depth = label_arr.shape[0]
                if label_depth < self.n_zSlices:
                    pad_total = self.n_zSlices - label_depth
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    label_arr = np.pad(
                        label_arr,
                        ((pad_before, pad_after), (0, 0), (0, 0)),
                        mode='constant', constant_values=0)
                if self.verbose:
                    print(f"Padded label along Z-axis from {label_depth} to {label_arr.shape[0]} slices")

            assert label_arr.shape == img_arr.shape, f"Segmentation and image shape mismatch for {img_name}: {label_arr.shape} vs {img_arr.shape}"

            mask = Utils3D.binary_mask(label_arr, keep_values=binary_mask).astype(img_arr.dtype)

            img_arr = img_arr * mask  # apply mask to extract ROI                

        # Resize volume to standard shape
        if self.resize:    
            if self.resize_method == "interpolation":      
                img_arr, _ = Utils3D.resize(img_arr, 
                                    desired_width=self.resize[2],
                                    desired_height=self.resize[1],
                                    desired_depth=self.resize[0],
                                    order=1,
                                    original_spacing=metadata["Spacing"])
            elif self.resize_method == "center_crop":
                img_arr = Utils3D.resize_with_center_crop(img_arr,
                                                          desired_width=self.resize[2],
                                                          desired_height=self.resize[1],
                                                          desired_depth=self.resize[0],
                                                          pad_value=self.resize_pad_value)
            else:
                raise ValueError("Unsupported keyword for resize. Supported are - interpolation and center_crop.")

            if self.verbose: print("Resized image shape in (D,H,W):", img_arr.shape)
            
        # Transforms (e.g., random flips, crops)
        if transform is not None:
            img_arr = transform(img_arr)       
         
        # Convert to torch tensor if it is a numpy array
        if isinstance(img_arr, np.ndarray): img_arr = torch.from_numpy(img_arr)
            
        # Convert MetaTensor -> plain tensor if needed
        if hasattr(img_arr, "as_tensor"): img_arr = img_arr.as_tensor()

        # Add channel dimension
        img_arr = img_arr.unsqueeze(0).float() # C x D x H x W
        
        return img_arr, img_name
        
    def __getitem__(self, idx):

        row = self.df.iloc[idx] # current row
        
        # Get label
        if self.classification_type == "multilabel":
            label = row[self.label_column].values.astype(np.float32)
            label = torch.from_numpy(label)
        else:
            # Handle if label column is provided in a list
            if isinstance(self.label_column, list):
                label = row[self.label_column[0]] # column name provided in list
            else:
                label = row[self.label_column] # column name provided in string

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
        
        # If dir_column is a string, then the dataloader acts as single-image loader. No need to iterate.
        if isinstance(self.dir_column, str): # single-image loader
            if self.verbose: print("dir_column is probided as string. So, loading as a single-image loader.")
            
            img_path = row[self.dir_column]

            # Prevent row[None], because it is invalid
            mask_path = None
            if isinstance(self.mask_column, str):
                mask_path = row[self.mask_column]
            
            # Make sure transform is not a list, if not None.
            if isinstance(self.transform, list): 
                self.transform = self.transform[0]

            img_arr, img_name = self._load_single_image(img_path=img_path, 
                                                   binary_mask=self.binary_mask if isinstance(self.binary_mask, (list, tuple)) else None, 
                                                   mask_path=mask_path,
                                                   transform=self.transform)
            
            return img_arr, label, img_name
        
        # If dir_column is a list, but length is i, then still the dataloader acts as single-image loader. 
        elif isinstance(self.dir_column, list) and len(self.dir_column) == 1: # still single-image loader
            if self.verbose: print("dir_column is probided as a list, but length is 1. So, loading as a single-image loader.")
            
            img_path = row[self.dir_column[0]]

            # Prevent row[None], because it is invalid
            mask_path = None
            if isinstance(self.mask_column, list):
                if self.mask_column[0] is not None: 
                    mask_path = row[self.mask_column[0]]
            elif isinstance(self.mask_column, str):
                mask_path = row[self.mask_column]

            # Make sure transform is not a list, if not None.
            if isinstance(self.transform, list): 
                self.transform = self.transform[0]

            img_arr, img_name = self._load_single_image(
                                                img_path=img_path,
                                                binary_mask=self.binary_mask if isinstance(self.binary_mask, (list, tuple)) else None,
                                                mask_path=mask_path,
                                                transform=self.transform)
                                            
            return img_arr, label, img_name
        
        else: # assuming multi-image dataloader
            if self.verbose: print("Loading as a multi-image loader.")
            
            # If binary_mask, mask_column, and transfrom are None, then repeat them to the length of dir_column
            if self.binary_mask is None: self.binary_mask = [None] * len(self.dir_column)
            if self.mask_column is None: self.mask_column = [None] * len(self.dir_column)
            
            # Do the same for transform, but it could be - None, True, False
            if self.transform is None or isinstance(self.transform, bool): self.transform = [self.transform] * len(self.dir_column)
            
            # Now, check if they all have the same length
            assert len(self.dir_column) == len(self.binary_mask), \
                f"Length of dir_column ({len(self.dir_column)}) and  binary_mask ({len(self.binary_mask)}) " \
                "in the config file should be the same for multi-image loader."
            
            assert len(self.binary_mask) == len(self.mask_column), \
                f"Length of binary_mask ({len(self.binary_mask)}) and mask_column ({len(self.mask_column)}) " \
                "in the config file should be the same for multi-image loader."    
                
            assert len(self.dir_column) == len(self.transform), \
                f"Length of dir_column ({len(self.dir_column)}) and  transform ({len(self.transform)}) " \
                "in the config file should be the same for multi-image loader."
            
            # Iterate over images
            img_arrs, img_names = [], []
            for dir_col, bin_mask, mask_col, transform in zip(self.dir_column, self.binary_mask, self.mask_column, self.transform):
                img_path = row[dir_col]
                
                # Prevent row[None], because it is invalid.
                mask_path = None
                if isinstance(mask_col, str):
                    mask_path = row[mask_col]
                              
                img_arr, img_name = self._load_single_image(img_path=img_path, 
                                                       binary_mask=bin_mask, 
                                                       mask_path=mask_path,
                                                       transform=transform)
                
                img_arrs.append(img_arr)
                img_names.append(img_name)
                
            return img_arrs, label, img_names
            
        
#%%
if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from transforms_v2 import Compose3D, Transform3D
    from transforms_v2 import monai_pipeline
    
    import pandas as pd

    "Uncomment for multi-label classification"
    # df = pd.read_excel('/research/m324371/Project/Digital_Twin/Classification/Dataframes/Dataset_791v2.xlsx') 
    # label_cols = ['Kidney cysts', 'Liver cysts', 'Diverticular disease']

    # # Transforms
    # transform_pipeline = Compose3D([
    #     (Transform3D.flip, {'axis': 'random'}, 0.5),  # 50% chance random flip
    #     (Transform3D.rotate, {'angle': 10, 'axes': (1, 2)}, 1.0),  # Always rotate 10 degrees along (1, 2)
    #     (Transform3D.center_crop, {'crop_size': (48, 96, 96), 'restore_shape': False, 'padding': False, 'interpolation_order':1}, 1.0)  # Always center crop
    # ])

    # dataset = ClsDataset(dataframe=df,
    #                     classification_type="multilabel",
    #                     dir_column="Directories",
    #                     label_column=label_cols,
    #                     onehot=False,
    #                     resample=(5.0,None,None), # (Sz,Sy,Sx) or (Sd,Sh,Sw)
    #                     n_zSlices=176,
    #                     zSlices_pad_value=500,
    #                     clip=(-1000,400),
    #                     normalize=True,
    #                     resize=(64, 128, 128), # (D,H,W)
    #                     transform=transform_pipeline, # None
    #                     verbose=True,
    #                     )

    # loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # # Example usage
    # for images, labels, image_names in loader:
    #     print(f"Image shape in (B,C,D,H,W): {images.shape}")  # [B, 1, D, H, W]
    #     print(labels.shape)  # [B, num_classes]
    #     print(labels)
    #     print(image_names)

    #     # Extract first image and convert to numpy
    #     volume = images[0, 0].cpu().numpy()  
        
    #     # Pass the extracted volume
    #     Utils3D.visualizer(volume, spacing=(1, 1, 1), cmap='gray', opacity="sigmoid")
    #     break

    "Uncomment for multi-class classification"
    df = pd.read_excel('/research/m324371/Project/Digital_Twin/Classification/Dataframes/AccRejRew_train_v2.xlsx') 

    # Transforms based on numpy and SciPy
    transform_pipeline = Compose3D([
        (Transform3D.flip, {'axis': 'random'}, 0.5),  # 50% chance random flip
        (Transform3D.rotate, {'angle': 10, 'axes': (1, 2)}, 1.0),  # Always rotate 10 degrees along (1, 2)
        (Transform3D.center_crop, {'crop_size': (48, 96, 96), 'restore_shape': False, 'padding': False, 'interpolation_order':1}, 1.0)  # Always center crop
    ])
    
    # Transform based on MONAI
    transform_dict = {
                    "RandFlip": {"spatial_axis": 0, "prob": 0.5},
                    "RandRotate": {"range_x": 0.52, "range_y": 0.52, "range_z": 0.52,
                                   "keep_size": True, "mode": "nearest", "prob": 0.5,}
                }
    monai_transform_pipeline = monai_pipeline(transform_dict)

    # Dataloader
    dataset = ClsDataset(dataframe=df,
                         dir_column=["Directories", "Directories"], # "Directories", 
                         label_column="Labels",
                         classification_type="multiclass",
                         binary_mask= [[1,2], None], # [1,2],
                         mask_column=["Seg_dirs", None],
                         onehot=True,
                         resample=(2.0, None, None), # (Sz,Sy,Sx) or (Sd,Sh,Sw)
                         n_zSlices=None, # 176,
                         zSlices_pad_value=None, # 500,
                         clip=None, # (-1000,400),
                         clip_percentile=(0.5, 99),
                         normalize=None, # True,
                         resize=(128, 256, 256), # (D,H,W)
                         resize_method="center_crop",
                         resize_pad_value=-1, 
                         transform=[None, monai_transform_pipeline], # None # transform_pipeline
                         verbose=True,
                        )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Example usage
    for images, labels, image_names in loader:
                
        print(labels.shape)  # [B, num_classes]
        print(labels)
        print(image_names) # dataloader will make it ("xyz",) as it performs batching.
        
        # # If it is a single-image loader: convert Tensor to list([Tensor]), so that we iterate
        # over both single-image and multi-image loader. 
        if isinstance(images, torch.Tensor): images = [images]

        for i, image in enumerate(images):
            print(f"Image shape in (B,C,D,H,W): {image.shape}")  # [B, 1, D, H, W]
            
            # Extract first image and convert to numpy
            volume = image[0, 0].cpu().numpy()  
    
            out_dir = "/research/m324371/Project/Digital_Twin/Classification/eGFR_prediction_resources/"
            out_name = os.path.splitext(image_names[i][0])[0] + f"_dloaderOut_{i}.nii.gz"
            out_path = os.path.join(out_dir, out_name)
            
            # Set some reasonable spacing (for visualization, this is fine)
            Utils3D.write_nifti(
                output_file=out_path,
                img_arr=volume,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0)
            )
    
            
            # # Pass the extracted volume
            # Utils3D.visualizer(volume, spacing=(1, 1, 1), cmap='gray', opacity="sigmoid")

        break            
        
        

    