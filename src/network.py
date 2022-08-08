import torch.nn as nn
import torch.nn.functional as f
import torch.optim


class Network(nn.Module):
    def __init__(self, dimension: int, hidden_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dimension, 30)] +
                                    [nn.Linear(30, 30)] * hidden_layers +
                                    [nn.Linear(30, 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = f.softplus(layer(x))
        return self.layers[-1](x)


class ReduceLROnDeviation:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 factor: float = 0.1,
                 interval: int = 10,
                 threshold: float = 0.2,
                 delay: int = 0,
                 cooldown: int = 0,
                 min_lr: float = 1e-8,
                 verbose: bool = True):
        self.param_groups = optimizer.param_groups
        self.factor = factor
        self.interval = interval
        self.threshold = threshold
        self.delay = delay
        self.cooldown = cooldown
        self.cooldown_count = 0
        self.min_lr = min_lr
        self.verbose = verbose
        self.epoch = -1
        self.device = self.param_groups[0]['params'][0].device
        self.history = torch.tensor([], device=self.device)
        self.msd = 0

    def step(self, loss):
        if self.is_skipped(): return
        if len(self.history) < self.interval:
            self.history = torch.cat((self.history, loss.unsqueeze(0)))
        else:
            history = self.history.clone()
            self.history = torch.tensor([], device=self.device)
            mean_loss = torch.mean(history)
            self.msd = (f.mse_loss(history, mean_loss.expand(self.interval)) / mean_loss ** 2).item()
            if self.msd > self.threshold and (history[1:] - history[:-1]).sum() > -1e-4:
                for i, group in enumerate(self.param_groups):
                    if (lr := group['lr']) - self.min_lr <= 1e-16: continue
                    group['lr'] = lr * self.factor
                    if self.verbose and self.param_groups[-1] is group:
                        print(f'Adjusting learning rate from {lr: .1e} to {group["lr"]: .1e}.')
                self.cooldown_count = self.cooldown
            return self.msd

    def is_skipped(self):
        self.epoch += 1
        if self.epoch < self.delay:
            return True
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return True
