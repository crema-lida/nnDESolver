import torch
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from time import time
import re
from itertools import permutations

from network import Network, ReduceLROnDeviation
from utils import generate_inputs, translate
from visual import Graph


class Equation:
    def __init__(self, eq, domain, step=0.01, targets=None):
        """
        Args:
            eq (str or tuple[str]): The left side of equation `F(x, u, u(1), ..., u(n)) = 0`, can either be
                str for a single equation or sequence of str for a system of equations.
            domain (tuple or dict[str, tuple]): The starting and ending values of independent variables.
                Specify a `dict` as {name: (start, end), ...} for multivariable functions.
            step (float or int): Optional. The gap between values within domain.
            targets (str or tuple[str]): Optional. str or sequence of str representing known solution(s)
                of the equation. They will be displayed on the graph if specified.
        """

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.domain = {'x': domain} if type(domain) == tuple else domain
        self.step = step
        self.inputs, self.var_names, self.inputs_shape = generate_inputs(self.domain, step)
        self.eq = [eq] if type(eq) == str else list(eq)
        self.targets = [targets] if type(targets) == str else list(targets) if targets else []
        self.functions = {}
        self.bc = []
        self.zeros = torch.zeros(1, 1, device=self.device)
        translate(self.functions, self.var_names, self.eq, self.targets)

    def boundary_condition(self, *bc):
        """
        Set or add to boundary conditions by passing a sequence of tuples or lists.
        Args:
            bc: A tuple or list defined as (x, value, order).
                `x` is the boundary position, can be either float or tuple of float.
                `value` is a floating-point number.
                `order` can be either a non-negative integer for ODEs or a string for PDEs,
                e.g. 'xx' or 'xyz'. If `order` is 0 or "", it denotes the function itself.
                For systems of differential equations, function names must be specified,
                e.g. 'u()', 'u(2)' or 'z(xx)'.
        """

        for item in bc:
            order = str(item[2]).strip()
            if order.endswith(')'):
                order = re.sub(r'(?<=\()\w*(?=\))', lambda match: f'"{match.group()}"', order)
                funcname = order[:order.index('(')]
            else:
                funcname = list(self.functions.keys())[0]
                order = funcname + f'("{order}")'
            pos = item[0] if type(item[0]) is tuple else (item[0],)
            pos = torch.tensor(pos, dtype=torch.float, device=self.device).unsqueeze(0).requires_grad_()
            target = torch.tensor([[item[1]]], dtype=torch.float, device=self.device)
            cond = self.Condition(pos, target, order, funcname)
            self.bc.append(cond)

    def solve(self, epoch: int = 10000, lr: float = 0.001, trivial_resist: bool = False):
        globs = {'torch': torch, 'np': np, 'Func': self.Function}
        for funcname in self.functions.keys():
            self.functions[funcname] = self.Function(funcname, self.var_names, self.device)
            exec(f'{funcname} = self.functions[funcname]')
            globs[funcname] = eval(funcname)
        self.Function.inputs = self.inputs
        targets = [eval(target, globs).numpy() for target in self.targets]
        update_graph = Graph(self.inputs.numpy(), *targets, shape=self.inputs_shape)

        index = torch.arange(self.inputs.size(0), dtype=torch.int32)
        dataset = TensorDataset(index, self.inputs)
        sample_size = index.size(0)
        batch_size = min(max(128, pow(2, int(np.log2(index.size(0) / 10)))), 131072)
        batch = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam([{'params': func.net.parameters()} for func in self.functions.values()], lr=lr)
        scheduler = ReduceLROnDeviation(optimizer)
        benchmark = {}
        boundary_loss = torch.tensor(0., device=self.device)

        print(f'computing device: {self.device}\n'
              f'sample size: {sample_size} | batch size: {batch_size}')
        start = time()

        # optimize boundary loss every epoch and equation loss every batch
        for cycle in range(epoch + 1):
            for bc in self.bc:
                self.Function.inputs = bc.position
                boundary_loss += f.mse_loss(eval(bc.order, globs), bc.target)
                eval(bc.funcname).derivatives.clear()
            if boundary_loss.grad_fn:
                boundary_loss.backward()
                scheduler.step(boundary_loss.detach_())
                benchmark['Boundary Loss'] = boundary_loss.item()
                boundary_loss.zero_()

            for _, (index, inputs) in enumerate(batch):
                self.Function.inputs = inputs.to(self.device).requires_grad_()
                zeros = self.zeros.expand(inputs.size(0), -1)

                if trivial_resist:
                    trivial_loss = list(map(lambda func: f.mse_loss(func(), zeros).clamp(1e-3).reciprocal(),
                                        self.functions.values()))
                    for value in trivial_loss:
                        if value > 1:
                            value.backward(retain_graph=True)
                    benchmark['Trivial Loss'] = sum(trivial_loss).item()

                eq_loss = sum(map(lambda eq: f.mse_loss(eval(eq, globs), zeros), self.eq))
                eq_loss.backward()
                benchmark['Equation Loss'] = eq_loss.item()

                outputs = {name: func().detach().cpu().numpy()
                           for name, func in self.functions.items()}
                update_graph(index.numpy(), outputs, cycle, benchmark)

                optimizer.step()
                optimizer.zero_grad()
                for func in self.functions.values():
                    func.derivatives.clear()

            if cycle % 500 == 0 or cycle == epoch:
                print(f'epoch {cycle} | elapsed: {time() - start: .2f} s | loss: {sum(benchmark.values()): .6e} | '
                      f'memory allocated: {torch.cuda.memory_allocated()}')
        update_graph.freeze()

    class Function:
        inputs: torch.Tensor

        def __init__(self, name, var_names, device):
            self.name = name
            self.var_names = var_names
            self.derivatives = {}
            self.vector = torch.ones(1, 1, device=device)
            self.net = Network(len(var_names)).to(device)

        def __call__(self, order='') -> torch.Tensor:
            if order.isdigit():
                order = 'x' * int(order)
            if order in self.derivatives:
                return self.derivatives[order]
            if order == '':
                self.derivatives[order] = self.net(self.inputs)  # compute zero-order derivative, i.e. the f(x) itself
            else:
                outputs = self(order[:-1])  # access lower order derivatives recursively
                vector = self.vector.expand(self.inputs.size(0), -1)
                result, = torch.autograd.grad(outputs,  # compute derivatives of `outputs` of its variables
                                              self.inputs,
                                              grad_outputs=vector,
                                              create_graph=True)
                # create a record of computed derivatives
                # presumes that all mixed partials exist and are continuous,
                # such that all mixed partials of a certain type are equal
                for i, name in enumerate(self.var_names):
                    for permutation in set(permutations(order[:-1] + name)):
                        self.derivatives[''.join(permutation)] = result[:, i:i + 1]
            return self.derivatives[order]

    class Condition:
        def __init__(self, position, target, order, funcname):
            self.position = position
            self.target = target
            self.order = order
            self.funcname = funcname


if __name__ == '__main__':
    sin, cos, tan, exp, ln, lg = np.sin, np.cos, np.tan, np.exp, np.log, np.log10
    e, pi = np.e, np.pi
    """
    NOTE:
        When editing math expressions in a string, the prefix `torch.` of math functions
        such as `torch.sin(x)` can be omitted for tensor operations. However, this is
        illegal for operations like `torch.exp(3.14)`. Please specify `np.exp(3.14)` for
        such plain floating-point calculations.
    """

    ode1 = Equation('x**2 * u(2) + x * u(1) + (x**2 - 0.25) * u()',
                    domain=(0.1, 1), step=0.001,
                    targets='(sin(x) + cos(x)) / sqrt(x)')
    ode1.boundary_condition((1, sin(1) + cos(1), 0),
                            (1, 0.5 * cos(1) - 1.5 * sin(1), 1))
    # ode1.solve()

    ode2 = Equation('u() * exp(x * u()) + cos(x) + x * exp(x * u()) * u(1)',
                    domain=(0.4, 8),
                    targets='log(2 - sin(x)) / x')
    ode2.boundary_condition((pi / 2, 0, 0))
    # ode2.solve()

    ode_system = Equation(('x(t) - y()',
                           'y(t) + x()'),
                          domain={'t': (-5, 5)},
                          targets=('cos(t) + sin(t)',
                                   'cos(t) - sin(t)'))
    ode_system.boundary_condition((0, 1, 'x()'),
                                  (2, cos(2) - sin(2), 'y()'))
    # ode_system.solve()

    pde1 = Equation('z() * z(y) + x * z(x) + z(xxx)',
                    {'x': (-2, 2), 'y': (-2, 2)}, 0.1)
    # pde1.solve()

    pde2 = Equation('S(tt) - (S(ti)**2 -1) / (S(ii) + S())',
                    {'t': (-5, 5), 'i': (-5, 5)}, 0.1)
    pde2.solve()
