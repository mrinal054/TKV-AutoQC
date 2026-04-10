import math
import torch

class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, total_epochs, verbose=False):
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.verbose = verbose

        def lr_lambda(epoch):
            # Linear warmup from min_lr to base_lr
            if epoch < warmup_epochs:
                warmup_ratio = epoch / float(max(1, warmup_epochs))
                return (min_lr / base_lr) + warmup_ratio * (1.0 - min_lr / base_lr)

            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / base_lr) + cosine_decay * (1.0 - min_lr / base_lr)

        super().__init__(optimizer, lr_lambda)

    def step(self, metrics=None):
        super().step()
        if self.verbose:
            print(f"[Scheduler] Epoch LR: {self.get_last_lr()}")

    def get_last_lr(self):
        return super().get_last_lr()


class CosineWarmupSchedulerNoBaseLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, min_lr, warmup_epochs, total_epochs, verbose=False):
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.verbose = verbose

        # Capture each group's initial LR to compute per-group min factors
        init_lrs = [pg["lr"] for pg in optimizer.param_groups]
        min_factors = [self.min_lr / max(1e-12, lr0) for lr0 in init_lrs]

        def make_lambda(min_factor):
            def lr_lambda(epoch):
                # Linear warmup from min_factor -> 1.0
                if epoch < self.warmup_epochs:
                    r = epoch / float(max(1, self.warmup_epochs))
                    return min_factor + r * (1.0 - min_factor)
                # Cosine decay 1.0 -> min_factor
                progress = (epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                c = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
                return min_factor + c * (1.0 - min_factor)
            return lr_lambda

        lambdas = [make_lambda(f) for f in min_factors]
        super().__init__(optimizer, lr_lambda=lambdas, verbose=verbose)

