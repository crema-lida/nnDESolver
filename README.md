# nnDESolver
Solve ordinary and partial differential equations with neural networks, implemented with PyTorch.
## Steps to Follow
**1. Write down your equation.**

Take Burgers' equation for example:
$$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=\frac{0.01}{\pi}\frac{\partial^2 u}{\partial x^2},\ x\in[-1,1],\ t\in[0,1]$$
The initial condition and Dirichlet boundary conditions read as:
$$u(0,x)=-\text{sin}(\pi x)$$
$$u(t,-1)=u(t,1)=0$$
We move all expressions of the above equations to the left side, so that we can use a lambda expression to describe them as:
```
from solver import Equation
from mathfunc import *

burgers = Equation(lambda u, t, x: (u('t') + u() * u('x') - 0.01 / pi * u('xx'),
                                    u(0, x) + sin(pi * x),
                                    u(t, -1),
                                    u(t, 1)),
                   t=(0, 1), x=(-1, 1),
                   step=(0.03, 0.005))
```
The first parameter of `Equation` receives a callable which **returns a tuple**. The callable should receive these parameters in sequence: the unknown function `u` and its variables `t, x`. Then, we use kwargs `t=(0, 1), x=(-1, 1)` to specify the domain of function `u`. We can also use `step` to specify the gap between values in each dimension (defaults to 0.01).

- If you hate lambda expressions, you might as well use the `def` keyword to define the callable outside and pass it to `Equation`.
- Remember to actually call `u()` to get its value, because `u` is literally a function.
- All math functions are from PyTorch. You can choose not to import all from `mathfunc.py`, but to use `torch.sin(x)` instead.

**2. Tweak the neural network. (Optional)**

In our case, we use 5 hidden layers in the network and display output data in a 2-D contour plot.
```
burgers.config(hidden_layers=5, graph='contour')
```
**3. Solve it!**
```
burgers.solve()
```
Run your code and see how things evolve.

![截图 2022-08-07 18-04-06](https://user-images.githubusercontent.com/100750226/183294657-1560b089-23b9-45e0-94d7-94a9c3f0fb99.png)

## Showcases
- The KdV equation

![截图 2022-08-07 19-04-31](https://user-images.githubusercontent.com/100750226/183295413-88133b58-323c-4869-92b8-a1b6fc1b3426.png)

- A system of ODEs

![截图 2022-08-07 19-38-11](https://user-images.githubusercontent.com/100750226/183295595-a268481d-c86d-4d67-a014-ef281ce70ad5.png)

All code examples are available in `examples.py`. Good luck!
