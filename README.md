# nnDESolver
Solve ordinary and partial differential equations with neural networks, based on PyTorch.
## Steps to Follow
- **Write down your equation.**
```
from equation import Equation

ode1 = Equation('x**2 * u(2) + x * u(1) + (x**2 - 0.25) * u()',
                domain=(0.01, 1), step=0.01,
                targets='(sin(x) + cos(x)) / sqrt(x)')
```
In the first parameter, "u()" represents the function of variable x, and "u(1)" or "u(x)" represents the first-order derivative of "u()", etc.

The parameter `targets` is optional. If you have a known solution, just specify it and see how close is output to target.

For systems of equations, replace the first parameter with a tuple or list of strings:
```
ode_system = Equation(('x(t) - y()',
                       'y(t) + x()'),
                      domain={'t': (-5, 5)},
                      targets=('cos(t) + sin(t)',
                               'cos(t) - sin(t)'))
```
- **Add boundary conditions**
```
ode1.boundary_condition((1, sin(1) + cos(1), 0),
                        (1, 0.5 * cos(1) - 1.5 * sin(1), 1))
```
Pass a sequence of tuples or lists consist of (position of boundary, value, order).

For PDEs, position of boundary is a tuple of numbers.
- **Solve it!**
```
ode1.solve()
```
Run your code and see how things evolve.
![图片](https://user-images.githubusercontent.com/100750226/179616883-f9885d66-e6dd-4af1-9f20-45751d3c30a5.png)

BTW, you can specify some parameters like:
```
ode1.solve(epoch=20000, lr=0.002, trivial_resist=True)
```
Specify `trivial_resist=True` if you don't want to see trivial solutions such as `y = 0`. However, this is not recommended for most cases.

An example for solving PDEs:
```
pde1 = Equation('y * z(x) + x * z(y)',
                {'x': (-2, 2), 'y': (-2, 2)})
pde1.solve()
```
![图片](https://user-images.githubusercontent.com/100750226/179619033-470dc1a0-efb6-4aba-891e-9f1a711f16e2.png)

More examples are available in the end of file `equation.py`. Good luck!
