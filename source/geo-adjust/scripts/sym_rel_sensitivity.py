from os.path import join

import numpy as np
import path
import sympy as sp
from numpy.linalg import inv

import pandas as pd

from examples.gm.trafo2d import gm_extended
from examples.gm.trafo2d_sensitivity import simulate_points
from geotrafo.helmert import helmert2d

####################################################################
# analyze relative sensitivity


def analyze_influence(s, factor):
    Sx = s._Sigma_xx
    A = s._A
    sigmas = {'full': Sx}
    for i, l in enumerate(s._sym_l):
        # update sigma
        S2 = s._Sigma_ll.copy()
        S2[i, i] *= (1-factor)
        Sx2 = inv(A.T @ inv(S2/s._var0_prio) @ A)
        sigmas[l.name] = Sx2*s._var0_post
        # WRONG!!!
        # Sx2 = inv(A.T @ inv(S2) @ A)
        # sigmas[l.name] = Sx2

    print('OK')
    S = np.zeros((4, 20))
    full = sigmas['full']
    cols = ' '.join(['yo_{0:02d} xo_{0:02d}'.format(i) for i in range(1,6)]).split() + \
           ' '.join(['yt_{0:02d} xt_{0:02d}'.format(i) for i in range(1,6)]).split()
    for i, c in enumerate(cols):
        S[0, i] = (1 - sigmas[c][10, 10] / full[10, 10]) / factor
        S[1, i] = (1 - sigmas[c][11, 11] / full[11, 11]) / factor
        S[2, i] = (1 - sigmas[c][12, 12] / full[12, 12]) / factor
        S[3, i] = (1 - sigmas[c][13, 13] / full[13, 13]) / factor

    obs_names = ['y{0}_u x{0}_u'.format(i) for i in range(1, 6)]
    obs_names += ['y{0}_t x{0}_t'.format(i) for i in range(1, 6)]
    obs_names = ' '.join(obs_names).split()

    params = ['dy', 'dx', 'm', 'alpha']
    sensitivity = pd.DataFrame(S,
                               index=params, columns=obs_names)

    distribution = pd.DataFrame(sensitivity.values / np.sum(sensitivity.values, 0),
                                index=params, columns=obs_names)

    contribution = pd.DataFrame(sensitivity.values / np.sum(sensitivity.values, 1)[:, None],
                                index=params, columns=obs_names)

    np.testing.assert_allclose(np.sum(distribution, 0), 1)
    np.testing.assert_allclose(np.sum(contribution, 1), 1)

    # HRs = pd.read_pickle(join('examples/data', 'HRs.pk'))
    # cs = []
    # for i in range(1, 5):
    #     # cs += [np.diag(HRs[i][0])[:, None]]
    #     cs += [np.sqrt(np.diag(HRs[i][0])/(np.diag(HRs[i][1])))[:, None]]
    #
    # H_sens = np.hstack(cs).T
    # H_df = pd.DataFrame(H_sens * 100, index=params, columns=obs_names)

    return sensitivity, distribution, contribution


def run():
    # simulate observations
    ur_sim, tps_sim, sigma = simulate_points()

    # centroid = np.mean(tps_sim, axis=0)
    # tps_sim = tps_sim - centroid
    #
    # centroid2 = np.mean(ur_sim, axis=0)
    # ur_sim = ur_sim - centroid2

    # adjust (simple classic)
    tp, ac, tr, rc = helmert2d(ur_sim, tps_sim)
    tp = tp[[0, 1, 3, 2]]
    tp[3] = tp[3] % (2 * np.pi) / np.pi * 200
    # adjust (simple)
    # result, solver = simple_gm(ur_sim, tps_sim, rot_approx=np.pi / 2)
    # adjust (extended)
    result_e, solver_e = gm_extended(ur_sim, tps_sim, rot_approx=0.,
                                     sigma_source=sigma[:10, :10], sigma_target=sigma[10:, 10:], load_functions=True)
    result_e.model.print_timing()
    # tp2 = result.x
    # tp2[3] = tp2[3] % (2 * np.pi) / np.pi * 200

    tp3 = result_e.x[-4:]
    tp3[3] = tp3[3] % (2 * np.pi) / np.pi * 200

    s, d, c = analyze_influence(solver_e, 0.9)
    return s, d, c


def sym():

    # # Not necessary but gives nice-looking latex output
    # # More info at: http://docs.sympy.org/latest/tutorial/printing.html
    # sp.init_printing()
    #
    # sx, sy, rho = sp.symbols('sigma_x sigma_y rho')
    # matrix = sp.Matrix([[sx ** 2, rho * sx * sy],
    #                      [rho * sx * sy, sy ** 2]])
    # print(sp.pretty(matrix.inv()))
    # print(sp.pretty(sp.simplify(matrix.inv())))

    print('lets go')

    i = sp.symbols('i')
    A = sp.Matrix(sp.symarray('a', (6, 3)))
    sigmas = sp.symbols(' '.join(['s{}'.format(k) for k in range(6)]))
    Sll = sp.diag(*sigmas)
    N = A.T * Sll.inv() * A
    # (sp.pretty(N))

    A2 = A.copy()
    Sll2 = sp.diag(*sigmas)
    # Sll2[0, 0] = Sll2[0, 0] * (1 - i)

    N2 = A.T * Sll2**-1 * A
    # print(sp.pretty(N2))

    Sxx = N.inv()
    Sxx2 = N2.inv()

    print('OK')


if __name__ == '__main__':

    s, d, c = run()

    s_sim = pd.read_pickle('examples/data/sensitivity_sim.pk')



    # sym()
    print('OK')