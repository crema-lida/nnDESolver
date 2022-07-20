import torch.nn as nn
import torch.nn.functional as f
import torch.optim


class Network(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dimension, 60)] +
                                    [nn.Linear(60, 60)] * 3 +
                                    [nn.Linear(60, 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = f.softplus(layer(x))
        return self.layers[-1](x)


class ReduceLROnDeviation:
    def __init__(self, optimizer: torch.optim.Optimizer,
                 factor: float = 0.1,
                 interval: int = 10,
                 threshold: float = 0.1,
                 delay: int = 0,
                 cooldown: int = 1000,
                 min_lr: float = 1e-8,
                 verbose: bool = True):
        self.optimizer = optimizer
        self.factor = factor
        self.interval = interval
        self.threshold = threshold
        self.delay = delay
        self.cooldown = cooldown
        self.cooldown_count = 0
        self.min_lr = min_lr
        self.verbose = verbose
        self.epoch = -1
        self.device = optimizer.param_groups[0]['params'][0].device
        self.history = torch.tensor([], device=self.device)
        self.msd = torch.tensor(0.)

    def step(self, loss):
        if self.is_skipped(loss): return

        if len(self.history) < self.interval:
            self.history = torch.cat((self.history, loss.detach().unsqueeze(0)))
        else:
            history = self.history.clone()
            self.history = torch.tensor([], device=self.device)

            mean_loss = torch.mean(history)
            self.msd = f.mse_loss(history, mean_loss.expand(self.interval)) / mean_loss ** 2
            if self.msd < self.threshold: return

            sample = history[::int(self.interval / 10)]
            slope_sum = (sample[1:] - sample[:-1]).sum()
            if slope_sum.abs() > 1e-4: return

            for i, group in enumerate(self.optimizer.param_groups):
                if (lr := group['lr']) - self.min_lr <= 1e-16: continue
                group['lr'] = lr * self.factor
                if self.verbose:
                    print(f'Adjusting learning rate from {lr: .1e} to {group["lr"]: .1e}.')
            self.cooldown_count = self.cooldown

    def is_skipped(self, loss):
        self.epoch += 1
        if self.epoch < self.delay or loss is None:
            return True
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return True
