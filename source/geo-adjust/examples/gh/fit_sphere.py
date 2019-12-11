import numpy as np
import numpy.linalg as la
import sympy as sp

from geo_adjust.gh import GHAdjust

points = np.array([
    [14.551, 9.495,  8.433],
    [ 9.071, 5.002, 10.892],
    [12.865, 4.306,  8.775],
    [ 9.170, 7.898, 11.930],
    [ 9.993, 9.575, 11.745],
    [12.152, 6.017, 11.054]
])


def get_sphere_approx(points):
    A = np.block([points, np.ones((points.shape[0], 1)) * 0.5])
    N = A.T @ A

    X_dach = la.inv(N) @ A.T @ (0.5 * np.sum(points ** 2, 1))
    X_dach[3] = np.sqrt(X_dach[3] + np.sum(X_dach[:3] ** 2))

    return X_dach


# x_0 = get_sphere_approx(points)
x_0 = [8, 5, 5, 4.5]

# create a solver object
solver = GHAdjust()

# the model
phi = sp.symbols('phi')

# observations (l)
x, y, z = sp.symbols('x y z')
# parameters (x)
xm, ym, zm, r = sp.symbols('xm ym zm r')

# fit the following exponential model (y = e^(mx+c))
phi = sp.Matrix([sp.sqrt((x - xm) ** 2 + (y - ym) ** 2 + (z - zm) ** 2) - r])
solver.add_model_autodiff(phi, (xm, ym, zm, r), (x, y, z))

solver.set_initial_params(x_0)
solver.add_data(points)

result = solver.solve()

# not really necessary (is conducted within AdjustmentResult)
solver._hauptprobe()
accept_H0, T, F = solver._globaltest(test='leq')
print(accept_H0, T, F)

result.parameters.print()

# x               :   9.99972450  +-  0.00105710
# y               :   7.99980653  +-  0.00053086
# z               :   6.99930612  +-  0.00158434
# r               :   5.00054199  +-  0.00141422



