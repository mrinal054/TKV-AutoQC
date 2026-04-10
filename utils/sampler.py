from torch.utils.data import Sampler
import numpy as np

class BalancedPerEpochSampler(Sampler):
    """
    Each epoch:
      - take all indices from minority_class (k samples)
      - sample k indices from each other class (without replacement)
      - shuffle indices
    """
    def __init__(self, labels, minority_class, seed=0):
        self.labels = np.asarray(labels)
        self.minority_class = int(minority_class)
        self.seed = seed
        self.epoch = 0

        # group indices by class
        self.class_to_indices = {}
        for c in np.unique(self.labels):
            c = int(c)
            self.class_to_indices[c] = np.where(self.labels == c)[0]

        self.minority_indices = self.class_to_indices[self.minority_class]
        self.k = len(self.minority_indices)  # 1300 in your case

        self.other_classes = [c for c in self.class_to_indices.keys() if c != self.minority_class]

        for c in self.other_classes:
            if len(self.class_to_indices[c]) < self.k:
                raise ValueError(f"Class {c} has {len(self.class_to_indices[c])} samples, need >= {self.k}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        chosen = [self.minority_indices]  # always all minority

        for c in self.other_classes:
            sampled = rng.choice(self.class_to_indices[c], size=self.k, replace=False)
            chosen.append(sampled)

        epoch_indices = np.concatenate(chosen)
        rng.shuffle(epoch_indices)

        return iter(epoch_indices.tolist())

    def __len__(self):
        return self.k * (1 + len(self.other_classes))
