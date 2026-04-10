"""
Author: Mrinal Kanti Dhar
January 2026        
"""

print('************ The code is loaded ************')

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Append paths
import sys
import os

sys.path.append(os.getcwd() + '/utils/') 
sys.path.append(os.getcwd() + '/losses/') 
sys.path.append(os.getcwd() + '/config/') 

#%% Imports
import dataloader
from transforms_v2 import monai_pipeline # for augmentation, importing from utils
# from networks import nets
import networks
from network_parameters.params import model_params 
from losses.loss import loss_func 

import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax, sigmoid
from torch.utils.data import DataLoader

from copy import deepcopy

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

from box import Box
import yaml
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import seaborn as sns

#%% Read config file
# Function to read config file from command line
def get_config_from_args():
    parser = argparse.ArgumentParser(description="Pass config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    return args
    
# Get the config file from command-line arguments
args = get_config_from_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
config = Box(config)

#%% Parameters
DEVICE = config.train.device
EPOCHS = config.train.epochs
BACKBONE_LR = config.train.backbone_lr #0.0001 # learning rate
CLASSIFIER_LR = config.train.classifier_lr
WEIGHT_DECAY = config.train.weight_decay #1e-5
SAVE_WEIGHTS_ONLY = config.train.save_weights_only
SAVE_BEST_MODEL = config.train.save_best_model
SAVE_LAST_MODEL = config.train.save_last_model
PERIOD = config.train.period # periodically save checkpoints
EARLY_STOP = config.train.early_stop
PATIENCE = config.train.patience # for early stopping
BATCH_SIZE = config.train.batch_size
N_CLASSES = config.train.n_classes
DIR_COLUMN = config.data.dir_column
LABEL_COLUMN = config.data.label_column
BINARY_MASK = config.data.binary_mask
MASK_COLUMN = config.data.mask_column
RESAMPLE = config.data.resample
N_ZSLICES = config.data.n_zSlices
Z_SLICES_PAD_VALUE = config.data.zSlices_pad_value
CLIP = config.data.clip
CLIP_PERCENTILE = config.data.clip_percentile
NORMALIZE = config.data.normalize
RESIZE = config.data.resize
RESIZE_METHOD = config.data.resize_method
RESIZE_PAD_VALUE = config.data.resize_pad_value
TRANSFORM = config.transform.do_transform
N_WORKERS = config.data.n_workers
METRIC_MODE = config.metric.metric_mode
ONE_HOT = config.train.one_hot
CLASSIFICATION_TYPE = config.classification_type
RESUME_TRAIN = config.train.retrain.resume_train
FREEZE_BACKBONE = config.model.freeze_backbone
UNFREEZE_EPOCH = config.model.unfreeze_epoch
USE_SAMPLER = config.data.use_sampler

#%% Directories
root = config.directories.root
result_dir = config.directories.result_dir

#%% Augmentation
if TRANSFORM: # if true, list all transform pipelines including None
    transform_list = []
    transform_keys = config.transform.transform_keys # e.g. ["transform1", "transform2"]
    for t_key in transform_keys:
        # Get transform dictionary 
        t_dict = getattr(config.transform, t_key) # format: Box({}) if not None
        
        if isinstance(t_dict, Box): t_dict = t_dict.to_dict() # remove Box and keep only the dictionary
        
        # Append transform pipeline
        if t_dict is None: transform_list.append(None) # no augmentation for this image
        else: transform_list.append(monai_pipeline(t_dict)) # append transform pipeline for this image
        
    transforms = transform_list # return all transforms
else:
    transforms = None # return None -> no augmentation

#%% Base model name
base_model_name = config.model.name + '_' + config.model.suffix
    
print("Base model name:", base_model_name)

#%% Read train and test dataframes
if config.phase == "train" or config.phase == "both":
    df_train = pd.read_excel(config.directories.excel_train_dir) 
    train_dirs = df_train[DIR_COLUMN[0]] if isinstance(DIR_COLUMN, list) else df_train[DIR_COLUMN] # df_train[DIR_COLUMN] # Full filepath of training images
    train_class = df_train[LABEL_COLUMN[0]] if isinstance(LABEL_COLUMN, list) else df_train[LABEL_COLUMN]
    print(df_train.head())

if config.phase == "test" or config.phase == "both":
    df_test = pd.read_excel(config.directories.excel_test_dir) 
    # test_dirs = df_test[DIR_COLUMN] # Full filepath of test images
    # test_class = df_test[LABEL_COLUMN[0]] if isinstance(LABEL_COLUMN, list) else df_test[LABEL_COLUMN]

#%% Helper function
# # Save model
# def save(model_path, epoch, model_state_dict, optimizer_state_dict):
#     "Save model"
#     state = {
#         'epoch': epoch + 1,
#         'state_dict': deepcopy(model_state_dict),
#         'optimizer': deepcopy(optimizer_state_dict),
#         }

#     torch.save(state, model_path)

# Move model state dict to CPU
def to_cpu_state_dict(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}

# Model optimizer state dict to CPU
def to_cpu_optimizer_state(opt_state):
    # opt_state is a dict with 'state' and 'param_groups'
    cpu_state = {"state": {}, "param_groups": opt_state["param_groups"]}
    for pid, st in opt_state["state"].items():
        cpu_state["state"][pid] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in st.items()
        }
    return cpu_state

def save(model_path, epoch, model, optimizer=None):
    state = {"epoch": epoch}
    state["state_dict"] = to_cpu_state_dict(model.state_dict())
    if optimizer is not None:
        state["optimizer"] = to_cpu_optimizer_state(optimizer.state_dict())
    torch.save(state, model_path)

# Decide when to unfreeze backbone
def unfreeze_now(freeze_backbone, unfreeze_epoch, epoch):
    if not freeze_backbone:
        return False     # backbone was never frozen

    if unfreeze_epoch is None:
        return False     # never unfreeze

    return epoch >= unfreeze_epoch # True or False based on the condition

# Calculate classwise accuracy
def classwise_accuracy(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    class_accuracies = []
    for i in range(gt.shape[1]):
        correct = (gt[:, i] == pred[:, i]).sum()
        acc = correct / gt.shape[0]
        class_accuracies.append(acc)
    return np.mean(class_accuracies), class_accuracies

# Calculate per-class specificity
def compute_per_class_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))  # shape (C, C)
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    return specificities

# Plot One-vs-Rest ROC curves
def plot_roc_curves(y_true_bin, y_score, n_classes, save_path):
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# Log per-class metrics in tensorboard
def log_per_class_metrics(writer, epoch_index, gt, pred, prob, label_names, phase):
    gt = np.array(gt)
    pred = np.array(pred)
    prob = np.array(prob)

    # Determine classification type
    is_multilabel = gt.ndim == 2 and gt.shape[1] > 1

    if is_multilabel:
        # Multi-label classification (gt and pred are [N, C])
        per_class_acc = [(gt[:, i] == pred[:, i]).sum() / gt.shape[0] for i in range(gt.shape[1])]
        per_class_prec = precision_score(gt, pred, average=None, zero_division=0)
        per_class_rec = recall_score(gt, pred, average=None, zero_division=0)
        per_class_f1 = f1_score(gt, pred, average=None, zero_division=0)
        try:
            per_class_auc = roc_auc_score(gt, prob, average=None)
            if isinstance(per_class_auc, float):
                per_class_auc = [per_class_auc]
        except ValueError:
            per_class_auc = [0.0] * gt.shape[1]
    else:
        # Multi-class classification (gt and pred are [N])
        num_classes = len(label_names)
        gt_bin = label_binarize(gt, classes=np.arange(num_classes))
        pred_bin = label_binarize(pred, classes=np.arange(num_classes))

        per_class_acc = [(gt == i).sum() and ((gt == i) & (pred == i)).sum() / (gt == i).sum() or 0 for i in range(num_classes)]
        per_class_prec = precision_score(gt, pred, average=None, zero_division=0)
        per_class_rec = recall_score(gt, pred, average=None, zero_division=0)
        per_class_f1 = f1_score(gt, pred, average=None, zero_division=0)
        try:
            per_class_auc = roc_auc_score(gt_bin, prob, average=None, multi_class='ovr')
        except ValueError:
            per_class_auc = [0.0] * num_classes

    for i, label in enumerate(label_names):
        writer.add_scalar(f'PerClass/Accuracy/{phase}/{label}', per_class_acc[i], epoch_index)
        writer.add_scalar(f'PerClass/Precision/{phase}/{label}', per_class_prec[i], epoch_index)
        writer.add_scalar(f'PerClass/Recall/{phase}/{label}', per_class_rec[i], epoch_index)
        writer.add_scalar(f'PerClass/F1/{phase}/{label}', per_class_f1[i], epoch_index)
        writer.add_scalar(f'PerClass/AUC/{phase}/{label}', per_class_auc[i], epoch_index)


#%% Function to run one epoch
# Link: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html    
def run_one_epoch(epoch_index, loader, model, loss_fn, optimizer=None, 
                  phase='train', metric_mode='macro', log_interval=10, 
                  writer=None, device='cuda'):
    
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    interval_loss_total = 0.0
    epoch_loss_total = 0.0
    valid_batches = 0

    epoch_gt, epoch_pred, epoch_prob = [], [], []
    interval_gt, interval_pred, interval_prob = [], [], []

    for i, (inputs, labels, _) in enumerate(loader):

        valid_batches += 1

        # Move inputs to device
        if isinstance(inputs, (list, tuple)):
            # multi-image case: list of tensors, e.g. [img1, img2, ...]
            inputs = [x.to(device) for x in inputs]
        else:
            # single-image case: single tensor
            inputs = inputs.to(device)

        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train): # controls whether autograd tracks operations
            outputs = model(inputs)

            # Ensure shapes match for binary (non-one-hot) case
            if CLASSIFICATION_TYPE == "binary" and not ONE_HOT:
                outputs = outputs.squeeze(-1) # squeeze model output [B,1] -> [B] (no effect if already [B])
                labels = labels.view(-1).float() # make sure labels are float and 1D
            elif CLASSIFICATION_TYPE == "multiclass" and not ONE_HOT:
                labels = labels.long()
            elif CLASSIFICATION_TYPE == "multilabel":
                labels = labels.float()

            # Calculate loss
            loss = loss_fn(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        loss_val = loss.item()
        interval_loss_total += loss_val
        epoch_loss_total += loss_val

        # Collect predictions and probabilities
        if CLASSIFICATION_TYPE == "binary":
            if ONE_HOT:                            # logits [B, 2]
                prob = torch.softmax(outputs, dim=1)
                pred = torch.argmax(prob, dim=1).cpu().numpy().tolist()
                lbls = torch.argmax(labels, dim=1).cpu().numpy().tolist()
                prob_pos = prob[:, 1].detach().cpu().numpy().tolist() # use positive-class prob for AUC
                prob_ = prob_pos
            else:                                   # logits [B]
                prob = torch.sigmoid(outputs)
                pred = (prob > 0.5).float().cpu().numpy().tolist()
                lbls = labels.cpu().numpy().tolist()
                prob_ = prob.detach().cpu().numpy().tolist()

        elif CLASSIFICATION_TYPE == "multiclass":   # logits [B, C]
            prob = torch.softmax(outputs, dim=1)                   # [B,C]
            lbls = (torch.argmax(labels, dim=1) if ONE_HOT else labels.long()).cpu().numpy().tolist()
            pred = torch.argmax(prob, dim=1).cpu().numpy().tolist()
            prob_ = prob.detach().cpu().numpy().tolist()           # store FULL matrix for AUC

        elif CLASSIFICATION_TYPE == "multilabel":   # logits [B, C]
            prob = torch.sigmoid(outputs)
            pred = (prob > 0.5).float().cpu().numpy().tolist()
            lbls = labels.cpu().numpy().tolist()
            prob_ = prob.detach().cpu().numpy().tolist()

        epoch_gt.extend(lbls)
        epoch_pred.extend(pred)
        epoch_prob.extend(prob_)

        interval_gt.extend(lbls)
        interval_pred.extend(pred)
        interval_prob.extend(prob_)

        if (i + 1) % log_interval == 0 and writer is not None:
            iter_loss = interval_loss_total / log_interval
            acc = accuracy_score(interval_gt, interval_pred)
            precision = precision_score(interval_gt, interval_pred, average=metric_mode, zero_division=0)
            recall = recall_score(interval_gt, interval_pred, average=metric_mode, zero_division=0)
            f1 = f1_score(interval_gt, interval_pred, average=metric_mode, zero_division=0)

            try:
                if CLASSIFICATION_TYPE == "binary":
                    auc = roc_auc_score(interval_gt, interval_prob)  # <-- interval_*
                elif CLASSIFICATION_TYPE == "multiclass":
                    auc = roc_auc_score(interval_gt, np.array(interval_prob), multi_class='ovr', average=metric_mode)
                else:  # multilabel
                    auc = roc_auc_score(np.array(interval_gt), np.array(interval_prob), average=metric_mode)
            except ValueError:
                auc = 0.0

            global_step = epoch_index * len(loader) + i + 1
            writer.add_scalar(f'Loss/{phase}_iter', iter_loss, global_step)
            writer.add_scalar(f'Accuracy/{phase}_iter_mean_classwise', acc, global_step)
            writer.add_scalar(f'Precision/{phase}_iter_{metric_mode}', precision, global_step)
            writer.add_scalar(f'Recall/{phase}_iter_{metric_mode}', recall, global_step)
            writer.add_scalar(f'F1/{phase}_iter_{metric_mode}', f1, global_step)
            writer.add_scalar(f'AUC/{phase}_iter_{metric_mode}', auc, global_step)

            interval_loss_total = 0.0
            interval_gt, interval_pred, interval_prob = [], [], []

    # Final metrics for the epoch
    avg_loss = epoch_loss_total / valid_batches if valid_batches > 0 else float('inf')
    epoch_gt = np.array(epoch_gt)
    epoch_pred = np.array(epoch_pred)
    epoch_prob = np.array(epoch_prob)

    acc = accuracy_score(epoch_gt, epoch_pred)
    precision = precision_score(epoch_gt, epoch_pred, average=metric_mode, zero_division=0)
    recall = recall_score(epoch_gt, epoch_pred, average=metric_mode, zero_division=0)
    f1 = f1_score(epoch_gt, epoch_pred, average=metric_mode, zero_division=0)
    try:
        if CLASSIFICATION_TYPE == "binary":
            auc = roc_auc_score(epoch_gt, epoch_prob)
        elif CLASSIFICATION_TYPE == "multiclass":
            auc = roc_auc_score(epoch_gt, epoch_prob, multi_class='ovr', average=metric_mode)
        else: # multilabel
            auc = roc_auc_score(epoch_gt, epoch_prob, average=metric_mode)
    except ValueError:
        auc = 0.0

    # Log final epoch metrics
    if writer is not None:
        writer.add_scalar(f'Loss/{phase}_epoch', avg_loss, epoch_index)
        writer.add_scalar(f'Accuracy/{phase}_epoch_mean_classwise', acc, epoch_index)
        writer.add_scalar(f'Precision/{phase}_epoch_{metric_mode}', precision, epoch_index)
        writer.add_scalar(f'Recall/{phase}_epoch_{metric_mode}', recall, epoch_index)
        writer.add_scalar(f'F1/{phase}_epoch_{metric_mode}', f1, epoch_index)
        writer.add_scalar(f'AUC/{phase}_epoch_{metric_mode}', auc, epoch_index)

        # Skip per-class logging for binary non-one-hot
        should_log_per_class = (CLASSIFICATION_TYPE != "binary") or ONE_HOT
        if should_log_per_class: # removed label_names from if condition
            label_names = [f"class_{i}" for i in range(N_CLASSES)]
            log_per_class_metrics(writer, epoch_index, epoch_gt, epoch_pred, epoch_prob, label_names, phase)

    if writer is not None: writer.flush()

    return avg_loss, acc, precision, recall, f1, auc


#%% Base model
# Dynamically get the model class from networks
get_model = getattr(networks, config.model.name) # get_model is a model class, not an object

params = model_params(config.model.name, config) # initialize the model with other parameters 

base_model = get_model(**params)

# print(base_model)  # To verify it's working

#%% Dynamically get dataloader
ClsDataset = getattr(dataloader, config.data.dataloader)

#%% Training

if config.phase == "train" or config.phase == "both":
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Prepare summary writer directory
    writer_dir = os.path.join(result_dir, base_model_name, 'logs')
    os.makedirs(writer_dir, exist_ok=True)

    # Save the config file
    with open(os.path.join(result_dir, base_model_name, "config_" + timestamp +".yaml"), "w") as file: # will put a new timestamp for resume training also
        yaml.dump(config.to_dict(), file, default_flow_style=False)
    
    # Create StratifiedKFold object
    list_for_val_result_df = [] # it will store the best results for validation
    list_for_val_preds_df = [] # it will store all validation predictions and probabilities
    
    # Prepare which list of folds to train. If run_folds in config file is None,
    # then generate a fold list [0,1,...,kfold-1].
    if config.train.run_folds is None:
        RUN_FOLDS = list(range(0, config.train.kfold))
    else:
        RUN_FOLDS = config.train.run_folds
        
    # Prepare a list of folds for resuming training
    if RESUME_TRAIN: RESUME_FOLDS = config.train.retrain.resume_folds # it is a list
        
    "Subplots for losses and lr for training and validation"
    if len(RUN_FOLDS) == 1:
        fig_loss, axs_loss = plt.subplots(1, 1, figsize=(4, 4))
        fig_lr, axs_lr = plt.subplots(1, 1, figsize=(4, 4))

        # Normalize axes to lists when there is only one fold
        axs_loss = [axs_loss]
        axs_lr = [axs_lr]
    else:
        fig_loss, axs_loss = plt.subplots(1, len(RUN_FOLDS), figsize=(18, 4))
        fig_lr, axs_lr = plt.subplots(1, len(RUN_FOLDS), figsize=(18, 4))

    axs_idx = 0 # set initial axis index to 0. So the first plot will be at 0th index.

    trainging_record = [] # Store training records for each fold 
    
    "StratifiedKFold"
    k = config.train.kfold  # Number of folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.train.kfold_seed)  

    "Loop through the folds"
    for fold, (train_index, val_index) in enumerate(skf.split(train_dirs, train_class)):
        # Check if it is in the run list.
        if fold in RUN_FOLDS:
        
            # Create model name
            kfold_model_name = f"fold_{fold}" 
            print(f"{kfold_model_name}\n", '-'*40)
    
            "Load model"
            # Make a deep copy of the base model
            model = deepcopy(base_model)
            model = model.to(DEVICE)

            # Freeze/unfreeze backbone
            if FREEZE_BACKBONE: 
                print("Freezing backbone ... ", end="")
                model.set_backbone_trainable(False)
                print("done!")
    
            "Optimizer"
            optimizer_name = config.train.get("optimizer", "adam").lower()  # default = "adam"
            
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(
                    [
                        {"params": model.backbone_parameters(), "lr": 0.0 if FREEZE_BACKBONE else BACKBONE_LR}, 
                        {"params": model.classifier_parameters(), "lr": CLASSIFIER_LR}, 
                    ],                   
                    weight_decay=WEIGHT_DECAY
                )
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.backbone_parameters(), "lr": 0.0 if FREEZE_BACKBONE else BACKBONE_LR}, 
                        {"params": model.classifier_parameters(), "lr": CLASSIFIER_LR}, 
                    ],                   
                    weight_decay=WEIGHT_DECAY
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            "Learning rate scheduler"
            scheduler_type = config.train.get("scheduler", {}).get("type", "reduce_on_plateau") # added by Abby
    
            if scheduler_type == "cosine_warmup":
                from schedulers import CosineWarmupScheduler
            
                warmup_epochs = config.train["scheduler"].get("warmup_epochs", 10)
                min_lr = config.train["scheduler"].get("min_lr", 1e-6)
                total_epochs = config.train.get("epochs", 100)
            
                scheduler = CosineWarmupScheduler(
                    optimizer,
                    base_lr=float(CLASSIFIER_LR),
                    min_lr=float(min_lr),
                    warmup_epochs=warmup_epochs,
                    total_epochs=total_epochs,
                )
            elif scheduler_type == "cosine_warmup_v2": # v2 does not require any base lr.
                from schedulers import CosineWarmupSchedulerNoBaseLR
            
                warmup_epochs = config.train["scheduler"].get("warmup_epochs", 10)
                min_lr = config.train["scheduler"].get("min_lr", 1e-6)
                total_epochs = config.train.get("epochs", 100)
            
                scheduler = CosineWarmupSchedulerNoBaseLR(
                    optimizer,
                    # base_lr=LR,
                    min_lr=min_lr,
                    warmup_epochs=warmup_epochs,
                    total_epochs=total_epochs,
                    verbose=True,
                )            
            else:
                # Default: keep ReduceLROnPlateau for all other cases (existing models)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.1,
                    mode='min',
                    patience=10,
                    min_lr=1e-5,
                    verbose=True,
                )
            
            "Loss function"
            if config.loss.weights is not None:
                weights = torch.tensor(config.loss.weights, dtype=torch.float32).cuda()
            else: weights = None
    
            loss_fn = loss_func(config.loss.name, weights)
            
            "Resume training"
            if RESUME_TRAIN:
                if fold in RESUME_FOLDS:
                    checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', kfold_model_name)

                    # Make sure that there is only one best model.
                    bestmodel_list = os.listdir(checkpoint_loc) 
                    assert len(bestmodel_list) == 1, \
                        f"Expected exactly 1 best model to resume training in {base_model_name}, but found {len(bestmodel_list)}."
                    
                    bestmodel_name = bestmodel_list[0] # format: best_model_{timestamp}.pth

                    # Restore previous timestamp
                    timestamp = bestmodel_name.replace("best_model_", "").replace(".pth", "") # only keep datetime

                    # Load checkpoint
                    checkpoint = torch.load(os.path.join(checkpoint_loc, bestmodel_name), map_location=DEVICE)
                    
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
                    
                    if 'epoch' in checkpoint: initial_epoch = checkpoint['epoch']
                    else: initial_epoch =  0
                    
                    print(f"***** Resuming training from {checkpoint_loc}")
                    print(f"***** Resuming training from epoch {initial_epoch}")
                else: 
                    pass
            else: # fresh training        
                # Create checkpoint directory
                checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', kfold_model_name)
                os.makedirs(checkpoint_loc, exist_ok=True)  
                
                initial_epoch =  0
            
            fold_writer_dir = os.path.join(writer_dir, f'fold_{fold}')
            writer = SummaryWriter(log_dir=fold_writer_dir)
            
            print('No. of training images:', len(train_index))
            print('No. of validation images:', len(val_index))
    
            # Create dataframes for this k-fold
            kfold_df_train = df_train.iloc[train_index]
            kfold_df_val = df_train.iloc[val_index]
        
            # Reset index. Start from 0
            kfold_df_train.reset_index(drop=True, inplace=True)
            kfold_df_val.reset_index(drop=True, inplace=True)
    
            "Create directories"
            # val_result_save_dir = os.path.join(result_dir, base_model_name, 'results_val', f"fold_{k}")
            save_fig_dir = os.path.join(result_dir, base_model_name, "plots")
    
            # os.makedirs(val_result_save_dir, exist_ok=True)
            os.makedirs(writer_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
    
            "Dataloader"
            train_dataset = ClsDataset(
                                        dataframe=kfold_df_train,
                                        dir_column=DIR_COLUMN,
                                        label_column=LABEL_COLUMN,
                                        classification_type=CLASSIFICATION_TYPE,
                                        binary_mask=BINARY_MASK,
                                        mask_column=MASK_COLUMN,
                                        onehot=ONE_HOT,
                                        resample=RESAMPLE,
                                        n_zSlices=N_ZSLICES,
                                        zSlices_pad_value=Z_SLICES_PAD_VALUE,
                                        clip=CLIP,
                                        clip_percentile=CLIP_PERCENTILE,
                                        normalize=NORMALIZE,
                                        resize=RESIZE, # (D,H,W)
                                        resize_method=RESIZE_METHOD,
                                        resize_pad_value=RESIZE_PAD_VALUE,
                                        transform=transforms, 
                                        verbose=False,
                                        )
    
            val_dataset = ClsDataset(
                                        dataframe=kfold_df_val,
                                        dir_column=DIR_COLUMN,
                                        label_column=LABEL_COLUMN,
                                        classification_type=CLASSIFICATION_TYPE,
                                        binary_mask=BINARY_MASK,
                                        mask_column=MASK_COLUMN,
                                        onehot=ONE_HOT,
                                        resample=RESAMPLE,
                                        n_zSlices=N_ZSLICES,
                                        zSlices_pad_value=Z_SLICES_PAD_VALUE,
                                        clip=CLIP,
                                        clip_percentile=CLIP_PERCENTILE,
                                        normalize=NORMALIZE,
                                        resize=RESIZE, # (D,H,W)
                                        resize_method=RESIZE_METHOD,
                                        resize_pad_value=RESIZE_PAD_VALUE,
                                        transform=None, 
                                        verbose=False,
                                        )
            
            # Perform sampling
            if USE_SAMPLER:
                print("Per-epoch balanced sampling is activated.")
                from sampler import BalancedPerEpochSampler # get sampler class from util

                # Labels must match the rows of kfold_df_train (NOT df_train!)
                train_labels = kfold_df_train[LABEL_COLUMN].values

                # Identify minority class
                class_counts = kfold_df_train[LABEL_COLUMN].value_counts()
                minority_class = int(class_counts.idxmin())     # class with minimum samples
                k = int(class_counts.min())                     # minority count (e.g., 1300)

                # print("Class counts in this fold:\n", class_counts.to_dict())
                # print("Minority class:", minority_class, "k =", k)

                sampler = BalancedPerEpochSampler(
                    labels=train_labels,
                    minority_class=minority_class,   # minority class label
                    seed=123
                )

            # Call dataloader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                sampler=sampler if USE_SAMPLER else None,
                shuffle=False if USE_SAMPLER else True, 
                num_workers=N_WORKERS)    
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=N_WORKERS)
    
            "Train N epochs"
            # Containers for validation
            val_gt, val_pred, val_names, val_prob = [], [], [], []
    
            start = time.time() # start of training
    
            best_vloss = 1_000_000.
            best_val_accuracy = 0.
            best_val_precision = 0.
            best_val_recall = 0.
            best_val_f1 = 0.
            best_val_auc = 0.
            save_model = False # initially it is False
            cnt_patience = 0
    
            store_train_loss, store_val_loss = [], []
            store_epochs = []
            best_model_epoch = initial_epoch # Initially, best epoch is the initial epoch. Useful for retrain.
                
            store_lrs = [] # store learning rate
            
            has_unfrozen = False # keep status of freezing
            
            for epoch in range(initial_epoch, EPOCHS):
                
                if USE_SAMPLER: sampler.set_epoch(epoch) # pass the current epoch
                
                print('EPOCH {}:'.format(epoch + 1))
    
                # Record learning rate used in this epoch
                lr_this_epoch = optimizer.param_groups[0]['lr']
                store_lrs.append(lr_this_epoch)
                if writer is not None:
                    writer.add_scalar('LR/epoch', lr_this_epoch, epoch)

                # Unfreeze backbone
                if (not has_unfrozen) and unfreeze_now(FREEZE_BACKBONE, UNFREEZE_EPOCH, epoch):
                    print("Unfreezing backbone ... ", end="")
                    # Set backbone parameters trainable
                    model.set_backbone_trainable(True)

                    # Update the backbone learning rate in the optimizer
                    optimizer.param_groups[0]["lr"] = BACKBONE_LR

                    # Only update cosine_warmup schedule as it requires base_lr. Others are fine.
                    if scheduler_type == "cosine_warmup":
                        # If the scheduler keeps per-group base LRs like PyTorch:
                        if hasattr(scheduler, "base_lrs"):
                            scheduler.base_lrs[0] = BACKBONE_LR
                        elif hasattr(scheduler, "base_lr"):
                            scheduler.base_lr = CLASSIFIER_LR  # or keep as is
                            
                    has_unfrozen = True # set it True so that next time it does not repeat unfreezing again

                    print("done!")
    
                # Training (Returns avg_loss, mean_acc, precision, recall, f1, auc)
                train_returns = run_one_epoch(
                    epoch_index=epoch,
                    loader=train_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    phase='train',
                    metric_mode=METRIC_MODE,
                    writer=writer,
                    log_interval=2,
                    device=DEVICE,
                    # label_names=LABEL_COLUMN
                )
    
                # Validation
                val_returns = run_one_epoch(
                    epoch_index=epoch,
                    loader=val_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=None,  # No update in val
                    phase='val',
                    metric_mode=METRIC_MODE,
                    writer=writer,
                    log_interval=2,
                    device=DEVICE,
                    # label_names=LABEL_COLUMN
                )
    
                # Store losses
                store_train_loss.append(train_returns[0]) # average loss is not a tensor
                store_val_loss.append(val_returns[0])
    
                print(f"Loss train {train_returns[0]:.4f} valid {val_returns[0]:.4f}, "
                    f"vMeanAccuracy: {val_returns[1]:.4f}, vPrecision: {val_returns[2]:.4f}, "
                    f"vRecall: {val_returns[3]:.4f}, vF1: {val_returns[4]:.4f}, vAUC: {val_returns[5]:.4f}")
    
                # Track best performance, and save the model's state
                if val_returns[0] < best_vloss:
                    best_vloss = val_returns[0]
                    print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
                    cnt_patience = 0 # reset patience
                    best_model_epoch = epoch
                    save_model = True
    
                else: cnt_patience += 1
    
                # Learning rate scheduler
                if scheduler_type == "cosine_warmup":
                    scheduler.step()  # no args, step every epoch
                else:
                    scheduler.step(val_returns[0])  # ReduceLROnPlateau uses val loss
    
                # Save the model
                if save_model:
                    # Add a timestamp to save best model. If the user forgets to provide a unique suffix, 
                    # it will overwrite the previous best_model if a timestamp is not provided. 
                    save(os.path.join(checkpoint_loc, 'best_model_' + timestamp + '.pth'),
                            epoch+1, model, optimizer) 
                    save_model = False
    
                # Early stopping
                if EARLY_STOP and cnt_patience >= PATIENCE:
                    print(f"Early stopping at epoch: {epoch:04d}")
                    break
    
                # Periodic checkpoint save
                if not SAVE_BEST_MODEL:
                    if (epoch+1) % PERIOD == 0:
                        save(os.path.join(checkpoint_loc, f"cp-{epoch+1:04d}.pth"), 
                             epoch+1, model, optimizer)
                        print(f'Checkpoint saved for epoch {epoch:04d}')
              
                # Print memory allocation
                a = torch.cuda.memory_allocated()/1024**3
                r = torch.cuda.memory_reserved()/1024**3
                print(f"GPU memory allocation: allocated={a:.2f} GB reserved={r:.2f} GB")

            if not EARLY_STOP and SAVE_LAST_MODEL:
                print('Saving last model')
                save(os.path.join(checkpoint_loc, 'last_model' + '.pth'),
                        epoch+1, model, optimizer)
    
            print('Best model epoch:', best_model_epoch)
            print('Min validation loss:', np.min(store_val_loss)) 
            end = time.time() # End of training
            exe_time = end - start
            print(f'Training time: {exe_time:.2f} seconds')
    
            "Plot"
            # Loss curves
            axs_loss[axs_idx].plot(store_train_loss, 'r')
            axs_loss[axs_idx].plot(store_val_loss, 'b')
            axs_loss[axs_idx].set_title(f"Loss (Fold {fold})")
            axs_loss[axs_idx].set_xlabel("Epoch")
            axs_loss[axs_idx].set_ylabel("Loss")
            axs_loss[axs_idx].legend(['train', 'val'])
        
            # Learning-rate curve
            # axs_lr[fold].plot(store_lrs) # uncomment for normal plot
            axs_lr[axs_idx].semilogy(store_lrs) # uncomment to plot semilog
            axs_lr[axs_idx].set_title(f"Learning Rate (Fold {fold})")
            axs_lr[axs_idx].set_xlabel("Epoch")
            axs_lr[axs_idx].set_ylabel("LR")
            axs_lr[axs_idx].legend(['lr'])
            
            "Store all metrics"
            trainging_record_dict = {
                "Model name": kfold_model_name,
                "Best epoch": best_model_epoch,
                "Min train loss": np.min(store_train_loss),
                "Min val loss": np.min(store_val_loss),
            }

            axs_idx += 1 # increment axs_idx

            trainging_record.append(trainging_record_dict)

            "Uncomment to clean-up after each fold"
            # print("Performing memory clean-up ... ", end="")
            # # Close the writer for this fold
            # if writer is not None:
            #     writer.close()

            # # Explicitly delete big objects
            # del model, optimizer, scheduler
            # del train_loader, val_loader
            # del train_dataset, val_dataset
            # del loss_fn

            # # Run garbage collector
            # gc.collect()

            # # Ask CUDA to release cached memory back to the driver
            # torch.cuda.empty_cache()
            
            # print("Done!")


    # Save plots
    fig_loss.tight_layout()
    fig_lr.tight_layout()

    fig_loss.savefig(os.path.join(save_fig_dir, f"{base_model_name}_loss.png"))
    fig_lr.savefig(os.path.join(save_fig_dir, f"{base_model_name}_lr.png"))

    # Save to excel
    trainging_record = pd.DataFrame(trainging_record)
    xl_path = os.path.join(result_dir, base_model_name, 'training_records_' + timestamp + '.xlsx')
    trainging_record.to_excel(xl_path, index=False)

    print("*"*20, "Training done", "*"*20)

else:
    print("*"*20, "Skipping training", "*"*20)

#%% Inference
if config.phase == "both" or config.phase == "test":
    
    if config.phase == "test":
        base_model_name = config.test.base_model_name

    # Dataloader
    test_dataset = ClsDataset(
                                dataframe=df_test,
                                dir_column=DIR_COLUMN,
                                label_column=LABEL_COLUMN,
                                classification_type=CLASSIFICATION_TYPE,
                                binary_mask=BINARY_MASK,
                                mask_column=MASK_COLUMN,
                                onehot=ONE_HOT,
                                resample=RESAMPLE,
                                n_zSlices=N_ZSLICES,
                                zSlices_pad_value=Z_SLICES_PAD_VALUE,
                                clip=CLIP,
                                clip_percentile=CLIP_PERCENTILE,
                                normalize=NORMALIZE,
                                resize=RESIZE, # (D,H,W)
                                resize_method=RESIZE_METHOD,
                                resize_pad_value=RESIZE_PAD_VALUE,
                                transform=None, 
                                verbose=False,
                                    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    # Test save directory
    test_save_dir = os.path.join(result_dir, base_model_name, 'results_test')
    os.makedirs(test_save_dir, exist_ok=True)

    # # Find the best model name from the k-fold summary report
    # excel_fullfile = os.path.join(result_dir, base_model_name, 'training_records.xlsx')
    
    # dataframe = pd.read_excel(excel_fullfile)
    # idx_min_val_loss = np.argmin(dataframe["Min val loss"])
    # best_model_name = dataframe.iloc[idx_min_val_loss]["Model name"]
    
    # print('Base model name:', base_model_name)
    # print('Best model index:', idx_min_val_loss)

    if config.test.type == "average" or config.test.type == "both": # currently, one supports 'average', not 'best'
        # Get all model names
        model_names = os.listdir(os.path.join(result_dir, base_model_name, "checkpoints")) # e.g. ["fold_0", "fold_1"]
        
        print("Evaluating on:", model_names)
        
        trained_models = []    

        for model_name in model_names:
            checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', model_name)

            # Make sure that there is only one best model.
            bestmodel_list = os.listdir(checkpoint_loc) 
            assert len(bestmodel_list) == 1, \
                f"Expected exactly 1 best model in {model_name}, but found {len(bestmodel_list)}."
            
            bestmodel_name = bestmodel_list[0] # format: best_model_{timestamp}.pth

            # Load checkpoint
            checkpoint = torch.load(os.path.join(checkpoint_loc, bestmodel_name), map_location=DEVICE)
        
            # Make a deep copy of the base model
            model_copy = deepcopy(base_model)
        
            # Load the weights into the copied model
            model_copy.load_state_dict(checkpoint['state_dict'])
            model_copy.eval()  # Set the copied model to evaluation mode
        
            # Append the copied model to the list of trained models
            trained_models.append(model_copy.to(DEVICE))
        
        print('No. of models:', len(trained_models))

    ##### Evaluation: Avg of all models

    # Pre-define type
    classification_type = config.classification_type

    true, pred, test_name, pred_probs = [], [], [], []

    with torch.no_grad():
        for i, (tinputs, tlabels, tnames) in tqdm(enumerate(test_loader)):
            # Move inputs to device
            if isinstance(tinputs, (list, tuple)):
                # multi-image case: list of tensors, e.g. [img1, img2, ...]
                tinputs = [x.to(DEVICE) for x in tinputs]
            else:
                # single-image case: single tensor
                tinputs = tinputs.to(DEVICE)
                
            tlabels = tlabels.to(DEVICE)

            # For multi-image, tnames is a list of names. So, keep the first one only.
            if isinstance(tnames, (list, tuple)): tnames = tnames[0]

            # Ensemble prediction
            store_pred = [model(tinputs) for model in trained_models]
            stacked_preds = torch.stack(store_pred)  # [n_models, B, C]
            toutputs = torch.mean(stacked_preds, dim=0)  # [B, C] or [B]

            # Handle classification and ONE_HOT 
            if classification_type == "binary":
                if ONE_HOT:
                    # logits [B,2], labels either [B,2] or [B]
                    test_prob = softmax(toutputs, dim=1)                   # [B,2]
                    pred_class = torch.argmax(test_prob, dim=1).cpu().numpy().astype(int).tolist()

                    # ensure labels are class indices
                    if tlabels.ndim > 1:  # one-hot
                        tlabels = torch.argmax(tlabels, dim=1)
                    tlabels = tlabels.cpu().numpy().astype(int).tolist()

                    # store positive class probabilities for AUC
                    pred_prob = test_prob[:, 1].detach().cpu().numpy().tolist()

                else:
                    # logits [B], labels [B]
                    test_prob = sigmoid(toutputs).detach()                 # [B]
                    pred_class = (test_prob > 0.5).cpu().numpy().astype(int).tolist()
                    tlabels = tlabels.cpu().numpy().astype(int).tolist()
                    pred_prob = test_prob.cpu().numpy().tolist()

            elif classification_type == "multiclass":
                # logits [B,C]
                test_prob = softmax(toutputs, dim=1).detach()              # [B,C]
                pred_class = torch.argmax(test_prob, dim=1).cpu().numpy().astype(int).tolist()

                # labels as class indices
                if ONE_HOT and tlabels.ndim > 1:
                    tlabels = torch.argmax(tlabels, dim=1)
                tlabels = tlabels.cpu().numpy().astype(int).tolist()

                # store FULL probability vector for AUC
                pred_prob = test_prob.cpu().numpy().tolist()

            else:
                raise ValueError("Unsupported classification_type. Must be 'binary' or 'multiclass'.")

            # Collect results
            true.extend(tlabels)
            pred.extend(pred_class)
            pred_probs.extend(pred_prob)
            test_name.extend(tnames)


    # Save predictions
    df_test_ = pd.DataFrame({
        "Names": test_name,
        "Label": true,
        "Prediction": pred,
        "Predicted Probability": pred_probs,
    })
    df_test_.to_excel(os.path.join(test_save_dir, base_model_name + "_avg.xlsx"))

    # Metrics
    y_true = df_test_["Label"].tolist()
    y_pred = df_test_["Prediction"].tolist()
    y_pred_probs = df_test_["Predicted Probability"].tolist()

    n_classes = len(set(y_true))
    
    if classification_type == "binary":
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc_value = roc_auc_score(y_true, y_pred_probs)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(test_save_dir, "roc_" + base_model_name + "_avg.png"))

    else:  # multiclass
        y_pred_prob_matrix = np.asarray(y_pred_probs)              # [N,C]
        C = y_pred_prob_matrix.shape[1]
        classes = list(range(C))

        cm = confusion_matrix(y_true, y_pred, labels=classes)

        specificity = None
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        accuracy = accuracy_score(y_true, y_pred)

    # binarize using consistent class list
        y_true_bin = label_binarize(y_true, classes=classes)
        try:
            auc_value = roc_auc_score(y_true_bin, y_pred_prob_matrix, multi_class='ovr', average='macro')
        except ValueError:
            auc_value = 0.0


        # ROC Curves per class
        plot_roc_curves(
            y_true_bin=y_true_bin,
            y_score=y_pred_prob_matrix,
            n_classes=C,
            save_path=os.path.join(test_save_dir, f"roc_multiclass_{base_model_name}_avg.png")
        )

    # Save metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc_value)

    test_results_df = pd.DataFrame({
        "Model name": ["ensemble_avg"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "AUC": [auc_value],
        "Specificity": [specificity if specificity is not None else np.nan],
    })
    test_results_df.to_excel(os.path.join(test_save_dir, base_model_name + "_metrics_avg.xlsx"))

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Pred {i}" for i in range(n_classes)],
                yticklabels=[f"True {i}" for i in range(n_classes)])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(test_save_dir, f"cmat_{base_model_name}_avg.png"))

    # Save classification report
    df_cr = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    df_cr.to_excel(os.path.join(test_save_dir, base_model_name + "_creport_avg.xlsx"))

    # Save per-class specificity
    per_class_specificity = compute_per_class_specificity(y_true, y_pred, n_classes)
    df_specificity = pd.DataFrame({
        "Class": list(range(n_classes)),
        "Specificity": per_class_specificity
    })
    df_specificity.to_excel(os.path.join(test_save_dir, base_model_name + "_per_class_specificity.xlsx"), index=False)

    print("*" * 20, "Avg. model evaluation done", "*" * 20)