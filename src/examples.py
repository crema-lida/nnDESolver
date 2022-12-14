import torch

if __name__ == '__main__':
    from solver import Equation
    import math
    from mathfunc import *

    ode1 = Equation(lambda u, x: (u() * exp(x * u()) + cos(x) + x * exp(x * u()) * u('x'),
                                  u(pi / 2)),
                    x=(0.5, 20),
                    exact_soln=lambda x: ln(2 - sin(x)) / x)
    # ode1.solve()

    ode_system = Equation(lambda x, y, t: (x('t') - y(),
                                           y('t') + x(),
                                           x(0) - 1,
                                           y(2) - math.cos(2) + math.sin(2)),
                          t=(-5, 5),
                          exact_soln=lambda t: (cos(t) + sin(t),
                                                cos(t) - sin(t)))
    # ode_system.solve()

    burgers = Equation(lambda u, t, x: (u('t') + u() * u('x') - 0.01 / pi * u('xx'),
                                        u(0, x) + sin(pi * x),
                                        u(t, -1),
                                        u(t, 1)),
                       t=(0, 1), x=(-1, 1), step=(0.05, 0.005))
    burgers.config(graph='contour')
    # burgers.solve()

    kdv = Equation(lambda u, t, x: (u('t') + u() * u('x') + u('xxx'),
                                    u(0, x) - 2 * cosh(x) ** (-2)),
                   t=(-5, 5), x=(-5, 5), step=0.1)
    kdv.config(graph='surface')
    # kdv.solve()

    tablet_boots_from_rest = Equation(lambda ux, t, y: (ux('t') - 2.98e-3 / 1000 * ux('yy'),
                                                        ux(0, y),
                                                        ux(t, 0) - 1,
                                                        ux(t, 10)),
                                      t=(0, 5), y=(0, 0.01), step=(1e-2, 1e-4))
    tablet_boots_from_rest.config(features=30)
    tablet_boots_from_rest.solve()

    # cannot solve equations below

    T_0 = 373.15
    T_f = 298.15
    T_0 -= T_f
    T_f = 0
    L = 0.01
    B = 0.001
    h = 18
    k = 0.0258
    Gamma = math.sqrt(h / (k * B))
    heat_transfer = Equation(lambda T, x: (T('xx') - Gamma ** 2 * (T() - T_f),
                                           T(0) - T_0,
                                           T('x', L)),
                             x=(0, L), step=0.0001,
                             exact_soln=lambda x: T_f + (T_0 - T_f) * (
                                     cosh(Gamma * x) - tanh(Gamma * x) * sinh(Gamma * x)
                             ))
    heat_transfer.config(features=30, hidden_layers=8)
    # heat_transfer.solve(epoch=20000)

    heat_transfer2d = Equation(lambda T, x, z: (T('xx') + T('zz'),
                                                T('z', x, 0) - h / k * (T(x, 0) - T_f),
                                                T('z', x, 2 * B) + h / k * (T(x, 2 * B) - T_f),
                                                T('z', x, B),
                                                T(0, z) - T_0,
                                                T('x', L, z)),
                               x=(0, L), z=(0, 2 * B), step=(L / 100, 2 * B / 100))
    heat_transfer2d.config(features=100, hidden_layers=8, cmap='jet')
    # heat_transfer2d.solve(lr=1e-3, epoch=50000)
