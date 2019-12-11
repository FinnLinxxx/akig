import numpy as np
from numpy.linalg import inv
import sympy as sp
import matplotlib.pyplot as plt

from geo_adjust.gh import GHAdjust

R = np.arange(0, 2 * np.pi, 0.01)

x_0 = [1.5, 1., 1., 2., np.pi]
a_0, b_0, my_0, mx_0, theta_0 = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]

x_true = [1.512, 1.03, 1.1, 2.1, -np.pi/4]
a_true, b_true, my_true, mx_true, theta_true = x_true[0], x_true[1], x_true[2], x_true[3], x_true[4]

xx_0 = mx_0 + a_0 * np.cos(R) * np.cos(theta_0) + b_0 * np.sin(R) * np.sin(theta_0)
yy_0 = my_0 - a_0 * np.cos(R) * np.sin(theta_0) + b_0 * np.sin(R) * np.cos(theta_0)

xx_true = mx_true + a_true * np.cos(R) * np.cos(theta_true) + b_true * np.sin(R) * np.sin(theta_true)
yy_true = my_true - a_true * np.cos(R) * np.sin(theta_true) + b_true * np.sin(R) * np.cos(theta_true)

y = yy_true + 0.1 * np.random.randn(len(R))
x = xx_true + 0.1 * np.random.randn(len(R))

plt.plot(y, x, '+', color='black', label='Measurements')
plt.axis('equal')

points = np.vstack((x, y)).T

plt.plot(yy_true, xx_true, label='True Ellipse')
plt.plot(yy_0, xx_0, label='Start Estimate Ellipse')
plt.legend()
plt.show()

# create a solver object
solver = GHAdjust()

# the model
phi = sp.symbols('phi')

# observations (l)
y, x, t = sp.symbols('y x t')
# parameters (x)
ym, xm, a, b, theta = sp.symbols('ym xm a b theta')

t = sp.atan((y - ym)/(x - xm))-theta

# fit the ellipse model (ax^2 + bxy + cy^2 + dx + ey + f = 0,)
phi = sp.Matrix([[xm + a * sp.cos(theta)*sp.cos(t) + b*sp.sin(theta)*sp.sin(t)],
                 [ym - a * sp.sin(theta)*sp.cos(t) + b*sp.cos(theta)*sp.sin(t)]])
solver.add_model_autodiff(phi, (ym, xm, a, b, theta), (y, x))

solver.set_initial_params([1.5, 1., 1., 2., np.pi])
solver.add_data(points)

result = solver.solve()

print(result.x)
print(result.sigma_x)

