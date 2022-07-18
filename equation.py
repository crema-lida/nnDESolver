import torch
import torch.nn.functional as f
import numpy as np
from time import time
import re
from itertools import permutations

from network import Network, ReduceLROnDeviation
from utils import generate_inputs, translate
from visual import graph


class Equation:
    def __init__(self, eq, domain: tuple | dict, step: float = 0.1, targets=None):
        """
        Args:
            eq: The left side of equation `F(x, u, u(1), ..., u(n)) = 0`, can either be `str`
                for a single equation or a sequence of strings for a system of equations.
            domain: The starting and ending values of independent variables. Specify a `dict`
                as {name: (start, end), ...} for multivariable functions.
            step (optional): The gap between values within domain.
            targets (optional): A string or sequence of strings representing known solution(s)
                of the equation. They will be displayed on the graph if specified.
        """

        self.device = torch.device('cuda:0')
        self.domain = {'x': domain} if type(domain) == tuple else domain
        self.step = step
        self.inputs, self.var_names, self.inputs_shape = generate_inputs(self.domain, step)
        self.eq = [eq] if type(eq) == str else list(eq)
        self.targets = [targets] if type(targets) == str else list(targets) if targets else []
        self.functions = {}
        self.bc = []
        self.zeros = torch.zeros_like(self.inputs[:, :1], device=self.device)
        translate(self.functions, self.var_names, self.eq, self.targets)

    def boundary_condition(self, *bc: tuple | list):
        """
        Set or add to boundary conditions by passing a sequence of tuples or lists.
        Args:
            bc:  A tuple or list defined as (x, value, order).

                `x` is the boundary position, can be either `float` or a `tuple` of floats.

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
            index = [0 for i in self.domain.keys()]
            position = item[0] if type(item[0]) == tuple else (item[0],)
            for i, limits in enumerate(self.domain.values()):
                index[i] = round((position[i] - limits[0]) / self.step)
            target = torch.tensor([item[1]], dtype=torch.float, device=self.device).squeeze(0)
            cond = self.Condition(index, target, order, funcname)
            self.bc.append(cond)

    def solve(self, epoch=30000, lr=0.001, trivial_resist=False):
        globs = {'self': self, 'torch': torch, 'np': np}
        for funcname in self.functions.keys():
            exec(f'{funcname} = self.Function(self.inputs, self.var_names, lr, self.device)')
            globs[funcname] = self.functions[funcname] = eval(funcname)
        targets = [eval(target, globs).detach() for target in self.targets]
        update_graph = graph(self.inputs.detach(), *targets, shape=self.inputs_shape)
        self.inputs = self.inputs.to(self.device).requires_grad_()

        benchmark = {}
        categories = ['Total Loss', 'Equation Loss', 'BC Loss']
        for name in categories:
            benchmark[name] = torch.tensor(0., device=self.device)

        start = time()
        for cycle in range(epoch + 1):
            for eq in self.eq:
                benchmark['Equation Loss'] += f.mse_loss(eval(eq, globs), self.zeros)
            for bc in self.bc:
                bc_loss = f.mse_loss(eval(bc.order, globs).view(self.inputs_shape)[bc.index], bc.target)
                benchmark['BC Loss'] += bc_loss
                eval(bc.funcname).bc_loss = bc_loss
            if trivial_resist:
                for func in self.functions.values():
                    trivial_loss = f.mse_loss(func(), self.zeros).clamp(1e-3).reciprocal() * 1e2
                    if trivial_loss > 1:
                        benchmark['Trivial Loss'] = trivial_loss
            for value in benchmark.values():
                benchmark['Total Loss'] += value
            benchmark['Total Loss'].backward()
            for func in self.functions.values():
                func.optimizer.step()
                func.scheduler.step(func.bc_loss)

            if cycle % 50 == 0:
                update_graph(cycle,
                             [func().detach().cpu() for func in self.functions.values()],
                             benchmark)
            if cycle % 1000 == 0:
                print(f'epoch {cycle} | loss: {benchmark["Total Loss"].data}')

            for func in self.functions.values():
                func.derivatives.clear()
                func.optimizer.zero_grad()
            for name in categories:
                benchmark[name] = benchmark[name].detach().zero_()
        end = time()
        print(f'CPU Time: {end - start} s')
        update_graph.freeze()

    class Function:
        def __init__(self, inputs, var_names, lr, device):
            self.inputs = inputs.to(device).requires_grad_()
            self.var_names = var_names
            self.derivatives = {}
            self.vector = torch.ones(inputs.size(0), 1, device=device)
            self.net = Network(inputs.size(1)).to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.scheduler = ReduceLROnDeviation(self.optimizer, delay=1000)
            self.bc_loss = None

        def __call__(self, order: str = '') -> torch.Tensor:
            if order.isdigit():
                order = 'x' * int(order)
            if order in self.derivatives:
                return self.derivatives[order]
            if order == '':
                self.derivatives[order] = self.net(self.inputs)  # compute zero-order derivative, i.e. the f(x) itself
            else:
                outputs = self(order[:-1])  # access lower order derivatives recursively
                result, = torch.autograd.grad(outputs,  # compute derivatives of `outputs` of its variables
                                              self.inputs,
                                              grad_outputs=self.vector,
                                              create_graph=True)
                # create a record of computed derivatives
                # presumes that all mixed partials exist and are continuous,
                # such that all mixed partials of a certain type are equal
                for i, name in enumerate(self.var_names):
                    for permutation in set(permutations(order[:-1] + name)):
                        self.derivatives[''.join(permutation)] = result[:, i:i + 1]
            return self.derivatives[order]

    class Condition:
        def __init__(self, index, target, order, funcname):
            self.index = tuple(index)
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
                    domain=(0.01, 1), step=0.01,
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
    # ode_system.solve(epoch=10000)

    pde1 = Equation('y * z(x) + x * z(y)',
                    {'x': (-2, 2), 'y': (-2, 2)},)
    pde1.solve()

    pde2 = Equation('S(tt) - (S(ti)**2 -1) / (S(ii) + S())',
                    {'t': (-5, 5), 'i': (-5, 5)})
    # pde2.solve(epoch=5000)
