import numpy as np
import sympy as sp
import numpy.linalg as la

from geo_adjust.config import GHModelConfig
from geo_adjust.gh import GHAdjust

# example from http://kisser.online/ausgleichung/ghm/kreis

points = np.array([[164.595, 73.414],
                   [159.396, 62.563],
                   [136.455, 45.842],
                   [112.648, 46.136],
                   [85.822, 71.995],
                   [84.701, 95.772],
                   [89.246, 106.901],
                   [113.501, 125.639],
                   [137.304, 125.374],
                   [164.847, 97.226]])

# points = np.array([[27.025, 15.790],
#                     [23.895, 18.102],
#                     [20.090, 18.150],
#                     [18.357, 15.068],
#                     [18.501, 11.649],
#                     [21.102, 10.108],
#                     [24.617, 9.771],
#                     [26.591, 12.756]])


def get_circle_approx(points):
    A = np.block([points, np.ones((points.shape[0], 1)) * 0.5])
    N = A.T @ A

    X_dach = la.inv(N) @ A.T @ (0.5 * np.sum(points ** 2, 1))
    X_dach[2] = np.sqrt(X_dach[2] + np.sum(X_dach[:2] ** 2))

    return X_dach


def extended_example():
    # do it with config
    gh_config = GHModelConfig()

    # observations (l)
    y, x = sp.symbols('y x')
    # parameters (x)
    xm, ym, r = sp.symbols('xm ym r')
    # the model
    phi = sp.Matrix([sp.sqrt((x - xm) ** 2 + (y - ym) ** 2) - r])

    gh_config.add_model_autodiff(phi, (ym, xm, r), (y, x))

    gh_config.save(r'../models/circle_gh.pk')

    del gh_config

    gh_config = GHModelConfig.load(r'../models/circle_gh.pk')

    # create the solver
    solver = GHAdjust(gh_config)

    # add initial approximations and data
    x_0 = get_circle_approx(points)
    solver.set_initial_params(x_0)
    solver.add_data(points)

    result = solver.solve()

    result.observations.set_obs_config([y, x],
                                       ['cm'] * 2,
                                       ['x', 'y'],
                                       cxs=[lambda x: x * 1e3] * 2,
                                       formats=['.2f'] * 2)

    result.model.print_timing()
    result.model.prints_info()
    result.parameters.print()
    result.observations.print()

    result.observations.plot()


if __name__ == '__main__':
    # test extended example
    extended_example()

