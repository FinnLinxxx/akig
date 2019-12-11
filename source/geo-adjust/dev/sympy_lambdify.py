import sympy as sp
import numpy as np

# Example of sympy and lambdify to produce A, B or other matrices/vectors

r, p, y, l, n, e, h = sp.symbols('r p y l n e h')
n = (sp.cos(r) * sp.sin(p) * sp.cos(y) + sp.sin(r) * sp.sin(y)) * l
e = (sp.cos(r) * sp.sin(p) * sp.sin(y) - sp.sin(r) * sp.cos(y)) * l
h = (sp.cos(r) * sp.cos(p)) * l

f_sym = sp.Matrix([n, e, h])
F_sym = f_sym.jacobian(sp.Matrix([r, p, y, l]))

# create a lambda expression for Jacoby matrix
f_mF = sp.lambdify((r, p, y, l), F_sym)

# create a lambda expression for model itself
f_mf = sp.lambdify((r, p, y, l), f_sym)

obs = np.random.rand(20, 4)

testF, testf = [], []
for l in obs:
    testF += [f_mF(l[0], l[1], l[2], l[3])]
    testf += [f_mf(l[0], l[1], l[2], l[3])]

F_mat = np.vstack(testF)
f_vec = np.vstack(testf)


print('Success')

