# Trainer
---
## trainer_v1_0
- Base training structure.
- Initial version used to establish core framework.
---

## trainer_v1_1
- Added **per-class analysis** during training.
- Introduced **retrain mode** for re-running selected folds or full training.
---

## trainer_v1_2
- Added support for **binary classification**.
- Added support for **multi-label classification** within the same framework.
---

## trainer_BcMcc_v1
- Supports both **binary** and **multiclass** classification.
- Added **k-fold cross-validation**.
- Supports **retrain**, allowing selective fold re-training.
---

## trainer_BcMcc_v2
- Adjusted **label tensor shapes** for consistent training/inference.
- Corrected **AUC computation** for binary and multiclass setups.
---

## trainer_BcMcc_v3
- **Optimizer** and **learning rate** now fully controlled via config file.
- Added **AdamW** optimizer option.
- Added **suffix naming** for organizing outputs.
- Added **weight unfreezing** functionality.
---

## trainer_BcMcc_v4
- **Dataloader is loaded dynamically** from the config file.
- **Class weights** are now passed to the loss function.
- Added **learning rate curve** visualization.
- Added **run_one_fold** option in the config for finer control over fold execution.
---

## trainer_multiImageNet_v1
- Designed for **multi-image models** (e.g., MultiImageNet).
- Uses [loaderv5](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/dataloader/loaderv5.py?ref_type=heads) as the multi-image dataloader.
- Here is a sample config file: [sample_config](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/config/template_trainer_multiImageNet.yaml?ref_type=heads).
- Supports **separate transforms** for each input image.
- Allows **freezing** and **scheduled unfreezing** of backbone.
- Provides full **k-fold CV**: sequential runs, multi-GPU fold distribution, or custom fold lists (e.g., `[1,3,5]`).
- Saves best checkpoints under:  
  `results/<unique folder name>/checkpoints/foldX/best_model_{timestamp}.pth`
- Supports **resume training**, including selective fold resume (e.g., `[1,2]`).
- Needs a **unique suffix** (e.g., `eGFR_001`) for output directories; missing suffix may cause issues during inference.
- Does not require `networks/net.py`; uses module-level `__init__.py` for network and dataloader creation.
---
