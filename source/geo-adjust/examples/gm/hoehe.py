from collections import Counter

import numpy as np
import sympy as sp

from geo_adjust.gm import GMAdjust, AdjustHelper


# Höhennetz von Benning (Kap. 8.1, p.242)
# bzw. http://web.archive.org/web/20171126020652/http://diegeodaeten.de:80/hoehennetzausgleichung_jag3d.html

def simple_try():
    from_pts = ['4', '5', '6', '1', '1', '2']
    to_pt = ['1', '2', '3', '2', '3', '3']
    measurements = np.array([[1.821, 1.720, 2.079, -0.097, -1.089, -0.995]]).T

    l_sym = ['dh_{}{}'.format(f, to_pt[i]) for i, f in enumerate(from_pts)]
    l_sym = sp.symbols(' '.join(l_sym))

    anschluss_pt = {'4': 82.0, '5': 82.002, '6': 80.651}
    neu_pt = {'1': 83.821, '2': 83.722, '3': 82.730}
    x0 = [83.821, 83.722, 82.730]
    c = [82.0, 82.002, 80.651]

    # Anschlusspunkte/Konstanten
    h4, h5, h6 = sp.symbols('g_4 g_5 g_6')
    # Neupunkte/parameters (x)
    h1, h2, h3 = sp.symbols('h_1 h_2 h_3')

    phi_hoehe = sp.Matrix([[h1 - h4], [h2 - h5], [h3 - h6], [h2 - h1], [h3 - h1], [h3 - h2]])

    solver = GMAdjust(max_iterations=100)
    solver.add_model_autodiff(phi_hoehe, (h1, h2, h3), l_sym, c=(h4, h5, h6))

    solver.set_initial_params(x0)
    solver.add_data(measurements.T)
    solver.add_constants(c)

    solver.set_stochastic_model(np.eye(len(measurements)) * 0.002 ** 2, var0=0.002 ** 2)

    result = solver.solve()

    result.parameters.print()
    result.observations.print()


def improved_try():
    pts = np.zeros((6,), dtype=[('pt', 'U10'), ('h', np.float64), ('fix', np.bool), ('datum', np.bool)])
    pts[:] = [('01', 83.821, False, False), ('02', 83.722, False, False), ('03', 82.730, False, False),
              ('04', 82.0, True, False), ('05', 82.002, True, False), ('06', 80.651, True, False)]

    obs = np.zeros((6,), dtype=[('from', 'U10'), ('to', 'U10'), ('dh', np.float64)])
    obs[:] = [('04', '01', 1.821), ('05', '02', 1.720), ('06', '03', 2.079),
              ('01', '02', -0.097), ('01', '03', -1.089), ('02', '03', -0.995)]

    helper = AdjustHelper()
    # create symbols for Anschlusspunkte (constants)
    anschluss_sym = helper.create_consts('g_{}', pts['pt'][pts['fix'] == True])
    # create symbols for Neupunkte (parameters)
    neu_sym = helper.create_params('h_{}', pts['pt'][pts['fix'] == False])

    obs_sym = helper.create_obs('dh_{0}{1}', obs[['from', 'to']].tolist())

    # define the height model
    dhij, hi, hj = sp.symbols('dh_ij h_i h_j')
    phi = sp.Matrix([hj - hi])

    phi2 = helper.create_model(phi, (hi, hj), obs[['from', 'to']].tolist())
    print('data')

    solver = GMAdjust(max_iterations=100)
    solver.add_model_autodiff(phi2, tuple(neu_sym), tuple(obs_sym), c=tuple(anschluss_sym))

    solver.set_initial_params(pts['h'][pts['fix'] == False].tolist())
    solver.add_data(obs['dh'])
    solver.add_constants(pts['h'][pts['fix'] == True].tolist())

    solver.set_stochastic_model(np.eye(len(obs['dh'])) * 0.002 ** 2, var0=0.002 ** 2)

    result = solver.solve()

    result.parameters.print()
    result.observations.print()


def improved_try2():
    pts = np.zeros((6,), dtype=[('pt', 'U10'), ('h', np.float64), ('fix', np.bool), ('datum', np.bool)])
    pts[:] = [('01', 83.821, False, False), ('02', 83.722, False, False), ('03', 82.730, False, False),
              ('04', 82.0, True, False), ('05', 82.002, True, False), ('06', 80.651, True, False)]

    obs = np.zeros((6,), dtype=[('from', 'U10'), ('to', 'U10'), ('dh', np.float64)])
    obs[:] = [('04', '01', 1.821), ('05', '02', 1.720), ('06', '03', 2.079),
              ('01', '02', -0.097), ('01', '03', -1.089), ('02', '03', -0.995)]

    # define the height model
    dhij, hi, hj = sp.symbols('dh_ij h_i h_j')
    phi = sp.Matrix([hj - hi])

    solver = AdjustHelper.add_block(pts, obs, phi, (hi, hj), ['from', 'to'], col_name='pt',
                                    template_const='g_{}', template_param='h_{}', template_obs='dh_{0}{1}',
                                    col_obs_name=['from', 'to'])

    solver.set_stochastic_model(np.eye(len(obs['dh'])) * 0.002 ** 2, var0=0.002 ** 2)
    result = solver.solve()

    result.parameters.print()
    result.observations.print()


def extended_try2():
    pts = np.zeros((6,), dtype=[('pt', 'U10'), ('h', np.float64), ('fix', np.bool), ('datum', np.bool)])
    pts[:] = [('01', 83.821, False, False), ('02', 83.722, False, False), ('03', 82.730, False, False),
              ('04', 82.0, True, False), ('05', 82.002, True, False), ('06', 80.651, True, False)]

    obs = np.zeros((18,), dtype=[('from', 'U10'), ('to', 'U10'), ('dh', np.float64)])
    obs[:] = [('04', '01', 1.821), ('05', '02', 1.720), ('06', '03', 2.079),
              ('01', '02', -0.097), ('01', '03', -1.089), ('02', '03', -0.995),
              ('04', '01', 1.821), ('05', '02', 1.720), ('06', '03', 2.079),
              ('01', '02', -0.097), ('01', '03', -1.089), ('02', '03', -0.995),
              ('04', '01', 1.821), ('05', '02', 1.720), ('06', '03', 2.079),
              ('04', '05', 0.002), ('04', '06', -1.349), ('05', '06', -1.351)
              ]
    obs['dh'] = obs['dh'] * 0.98

    # define the height model
    dhij, hi, hj, m = sp.symbols('dh_ij h_i h_j m')
    phi = sp.Matrix([m * (hj - hi)])

    solver = AdjustHelper.add_block(pts, obs, phi, (hi, hj), ['from', 'to'], col_name='pt',
                                    template_const='g_{}', template_param='h_{}', template_obs='dh_{0}{1}',
                                    col_obs_name=['from', 'to'], extra_params=[m], extra_params_initial=[1.])

    solver.set_stochastic_model(np.eye(len(obs['dh'])) * 0.002 ** 2, var0=0.002 ** 2)
    result = solver.solve()

    result.model.print_info()
    result.parameters.print()
    result.observations.print()


if __name__ == '__main__':
    # simple_try()
    improved_try2()
    # extended_try2()

    print('Success')
