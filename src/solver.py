import torch
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
from time import time, sleep
from itertools import permutations
from threading import Thread
from typing import Union

from network import Network, ReduceLROnDeviation
from utils import parse_inputs, gen_inputs, downsample, get_bdy_pos
from visual import Graph


class Equation:
    def __init__(self,
                 equations: type(lambda: ...),
                 step: Union[float, tuple[float, ...]] = 0.01,
                 exact_soln: type(lambda: ...) = None,
                 **domain: tuple[float, float]):
        self.inputs, coords = gen_inputs(domain, step)
        self.indices, self.ds_coords = downsample(self.inputs, coords)  # data will be down-sampled to plot graph
        self.var_names = coords.keys()
        self.functions = {name: None for name in equations.__code__.co_varnames[:-len(self.var_names)]}
        self.equations = equations
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.exact_soln = exact_soln
        self.features = 10
        self.hidden_layers = 5
        self.graph = 'contour'
        self.cmap = 'viridis'
        self.init_config = None

    def solve(self, epoch: int = 5000, lr: float = 1e-3):
        sample_size = self.inputs.shape[0]
        dataset = TensorDataset(torch.from_numpy(self.indices), torch.from_numpy(self.inputs))
        batch_size = min(max(256, pow(2, int(np.log2(sample_size / 10)))), 131072)
        batch = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        for name in self.functions.keys():
            self.functions[name] = self.Function(name, self.var_names, self.device, self.features, self.hidden_layers)
        optimizer = torch.optim.Adam([{'params': func.net.parameters()} for func in self.functions.values()], lr=lr)
        scheduler = ReduceLROnDeviation(optimizer)
        benchmark = {'Loss': 0}
        zero = torch.zeros(1, 1, device=self.device)
        range_vars = range(len(self.var_names))
        solns = self.exact_soln(*map(lambda arr: torch.from_numpy(arr),
                                     np.meshgrid(*self.ds_coords.values()))) if self.exact_soln else ()
        solns = solns if type(solns) is tuple else (solns,)
        update_graph = Graph(self.ds_coords, *map(lambda arr: arr.numpy(), solns), graph=self.graph, cmap=self.cmap)

        def refresh_status(stat):
            start = time()
            max_step = math.ceil(sample_size / batch_size)
            while True:
                sleep(0.3)
                print(f'\repoch {stat["epoch"]}/{epoch} | step {stat["step"]}/{max_step} | '
                      f'elapsed: {time() - start: .2f} s | loss: {benchmark["Loss"]: .6e}', end='')

        print(f'computing device: {self.device}\n'
              f'sample size: {sample_size} | batch size: {batch_size}')
        stat = {'epoch': 0, 'step': 0,
                'outputs': {name: np.full((self.indices[self.indices != -1].size, 1), np.nan)
                            for name in self.functions.keys()}}
        updater = Thread(target=refresh_status, args=(stat,), daemon=True)
        updater.start()

        for cycle in range(1, epoch + 1):
            stat['epoch'] = cycle

            for step, (indices, inputs) in enumerate(batch, 1):
                stat['step'] = step

                self.Function.inputs = inputs.to(self.device).requires_grad_()
                equations = self.equations(*self.functions.values(),
                                           *map(lambda i: self.Function.inputs[..., i: i + 1], range_vars))
                loss = sum(map(lambda output: f.mse_loss(output, zero.expand(output.size(0), -1)), equations))
                loss.backward()
                for group in optimizer.param_groups:
                    clip_grad_norm_(group['params'], max_norm=0.5, error_if_nonfinite=True)

                choice = indices != -1
                indices = indices.numpy()[choice]
                for name, func in self.functions.items():
                    stat['outputs'][name][indices] = func().detach().cpu().numpy()[choice]

                optimizer.step()
                benchmark['MSD'] = scheduler.step(loss.detach_())
                benchmark['Loss'] = loss.item()
                optimizer.zero_grad()
                for func in self.functions.values():
                    func.derivatives.clear()

            update_graph(stat['outputs'], stat['epoch'], benchmark)

    class Function:
        inputs: torch.Tensor

        def __init__(self, name, var_names, device, features, hidden_layers):
            self.name = name
            self.var_names = var_names
            self.device = device
            self.derivatives = {}
            self.vector = torch.ones(1, 1, device=device)
            self.net = Network(len(var_names), features, hidden_layers).to(device)

        def __call__(self, *args, inputs=None) -> torch.Tensor:
            """
            Args:
                order (Optional): str. Order of the derivative.
                position (Optional): number, Tensor. Where to apply the boundary condition.
            """
            order, position = parse_inputs(args)
            if position:
                inputs = get_bdy_pos(position, self.device)
            elif inputs is None:
                inputs = self.inputs
            if (inp := id(inputs)) not in self.derivatives:
                self.derivatives[inp] = {}
            if order in self.derivatives[inp]:
                return self.derivatives[inp][order]
            if order == '':
                self.derivatives[inp][order] = self.net(inputs)  # compute zero-order derivative, i.e. the outputs
            else:
                outputs = self(order[:-1], inputs=inputs)  # access lower order derivatives recursively
                vector = self.vector.expand(inputs.size(0), -1)
                result, = torch.autograd.grad(outputs,  # compute derivatives of `outputs` of its variables
                                              inputs,
                                              grad_outputs=vector,
                                              create_graph=True)
                # save computed derivatives.
                # presumes that all mixed partials exist and are continuous,
                # such that all mixed partials of a certain type are equal.
                for i, name in enumerate(self.var_names):
                    for permut in set(permutations(order[:-1] + name)):
                        self.derivatives[inp][''.join(permut)] = result[:, i:i + 1]
            return self.derivatives[inp][order]

    def config(self, **kwargs):  # TODO initial configuration
        if 'features' in kwargs:
            self.features = kwargs['features']
        if 'hidden_layers' in kwargs:
            self.hidden_layers = kwargs['hidden_layers']
        if 'graph' in kwargs:
            self.graph = kwargs['graph']
        if 'cmap' in kwargs:
            self.cmap = kwargs['cmap']
        if 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
