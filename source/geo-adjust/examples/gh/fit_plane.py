import numpy as np
import sympy as sp

from geo_adjust.config import GHModelConfig
from geo_adjust.gh import GHAdjust

# points = np.array([[-10., -15., -20.],
#                    [-1., -0.5, -2.],
#                    [1., 1.5, 1.],
#                    [10., 15., 20.]])

points = np.array([[96.993, 42.615, 12.749],
    [97.320, 44.299, 13.899],
    [97.451, 43.588, 13.487],
    [96.497, 43.323, 13.098],
    [96.704, 42.596, 12.605],
    [96.859, 42.915, 12.838]])


def get_plane_approx(points):
    pass


if __name__ == '__main__':

    # x0 = [-0.8, 0.1, 0.3, 0.]
    # x0 = [0.587140936123470, 0.742845397385766, 0.321630590446305, 106.974067580262968]
    x0 = [0.29267, 0.11019, 0.94984, 106.97407]
    # x0 = [-0.9, 0.15, 0.32, 0.1]

    config = GHModelConfig()

    # the model
    phi, gamma = sp.symbols('phi gamma')

    # observations (l)
    x, y, z = sp.symbols('x y z')
    # parameters (x)
    a, b, c, d = sp.symbols('a b c d')

    # fit the plane equation with constraint
    phi = sp.Matrix([a * x + b * y + c * z + d])
    gamma = sp.Matrix([sp.sqrt(a ** 2 + b ** 2 + c ** 2) - 1])

    config.add_model_autodiff(phi, (a, b, c, d), (x, y, z))
    config.add_constraint_autodiff(gamma)


    # create a solver object
    solver = GHAdjust(config, max_iterations=100)
    # solver.set_options(max_iterations=100)

    solver.set_initial_params(x0)
    solver.add_data(points)

    result = solver.solve()

    # nx              :  -0.19479704  +-  0.05157134
    # ny              :  -0.54492936  +-  0.02372937
    # nz              :   0.81554037  +-  0.01040697
    # d               :  31.74898960  +-  4.30808159

    result.model.print_timing()
    result.model.print_info(print_matrices=True)
    result.model.print_summary()
    result.parameters.print(print_correlation=False)
    result.parameters.plot_corr()
    result.observations.print()
