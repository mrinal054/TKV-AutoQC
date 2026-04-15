# [MultiImageNet](https://gitlab.mayo.edu/kline-lab/dtcls_repo/-/blob/main/networks/multiImageNet.py?ref_type=heads)

MultiImageNet is a flexible **multi-backbone classification model** designed for 2D and 3D medical imaging tasks.  
It processes multiple input images through multiple feature extractors (backbones), concatenates their feature vectors, and performs final prediction using a unified classification head.

This architecture is ideal for:
- Multi-view learning  
- Image + masked image fusion  
- Multi-phase CT  
- Multi-modal MRI  
- Any pipeline requiring **feature-level fusion** from multiple models

---

## What MultiImageNet Does

- Dynamically loads **one or more backbones** based on a YAML configuration file.
- Each backbone produces a feature vector of size **[B, C]**.
- Feature vectors from all backbones are concatenated → **[B, ΣC]**.
- A fully connected **classification head** produces the final class logits.
- Supports:
  - Arbitrary number of backbones
  - 2D or 3D input volumes
  - Per-backbone freezing/unfreezing
  - Multi-image input (one image per backbone)
  - Separate LR scheduling for backbones vs classifier

---

## Requirements

### 1. YAML configuration file
The YAML must define:
- Backbone names
- Their class names (as defined in `networks/__init__.py`)
- Their initialization arguments

**All backbones must return a pooled + flattened feature vector** of shape: [B, C]`

### 2. `networks` package
The YAML `modelclass` names must exist in:

```
networks/__init__.py
```

### 3. Classification head
The model uses:

```
classification_head.ClassificationHeadWithoutFlatten
```

This must be importable and compatible in your environment.

### 4. Dummy input size
Feature dimensions are inferred using `dummy_size`, such as:

```
[1, 1, 96, 96, 96]  # for 3D inputs
```

Backbones must accept this shape.

---

## Constructor

```python
model = MultiImageNet(
    config_path="networks/config_multiImageNet.yaml",
    n_classes=3,
    out_channels=[32, 16],
    dropout=0.1,
    device="cuda",
    dummy_size=[1, 1, 96, 96, 96],
    verbose=True,
)
```

### Key Arguments

- **config_path** — path to YAML file defining backbones  
- **n_classes** — number of output classes  
- **out_channels** — hidden-layer channel sizes for the classifier  
  - First channel size is auto-set to `sum(feature_dims)`  
- **dropout** — dropout probability  
- **device** — "cuda" or "cpu"  
- **dummy_size** — used to infer backbone output channels  
- **verbose** — prints backbone shapes and loading info  

---

## Forward Usage

### Multi-backbone (multi-input) example

```python
import torch
import networks

x1 = torch.randn(2, 1, 96, 96, 96)
x2 = torch.randn(2, 1, 96, 96, 96)

MultiImageNetCls = getattr(networks, "MultiImageNet")

model = MultiImageNetCls(
    config_path="networks/config_multiImageNet.yaml",
    n_classes=3,
    out_channels=[32, 16],
    dropout=0.1,
    device="cpu",
    verbose=True,
)

out = model([x1, x2])
print(out.shape)   # e.g., torch.Size([2, 3])
```

### Single-backbone usage

```python
x = torch.randn(4, 1, 96, 96, 96)

model = MultiImageNet(
    config_path="networks/config_singleBackbone.yaml",
    n_classes=2,
    out_channels=[64],
    dropout=0.2,
)

logits = model(x)
print(logits.shape)  # [4, 2]
```

---

## Training Tips

### Separate learning rates for backbones vs classifier

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone_parameters(), "lr": 1e-4},
    {"params": model.classifier_parameters(), "lr": 1e-3},
])
```

### Freezing and unfreezing backbones

Freeze:

```python
model.set_backbone_trainable(False)
```

Unfreeze later:

```python
model.set_backbone_trainable(True)
```

Useful for:
- Transfer learning  
- Linear probing  
- Gradual unfreezing strategies  

---

## Input Requirements

`multi_images` in `forward()` must be:

- **list/tuple** containing one image per backbone, or  
- **single Tensor** only if exactly one backbone is defined  

MultiImageNet will assert if:
- Number of tensors ≠ number of backbones  
- Wrong data type is passed  
- Shapes are incompatible with backbone definitions  

---

## Summary

### ✔ What it does
- Dynamically creates a multi-backbone classifier  
- Processes multiple input images  
- Concatenates backbone features  
- Performs final prediction with a fully connected head  

### ✔ What it requires
- YAML file defining backbones  
- Valid backbone classes in `networks/nets.py`  
- Classification head module  
- Dummy input shape that your backbones accept  

### ✔ How to use
- Load model with YAML config  
- Pass images as `[image1, image2, ...]`  
- Optionally freeze/unfreeze backbones  
- Use separate optimizers/LR groups if desired  
