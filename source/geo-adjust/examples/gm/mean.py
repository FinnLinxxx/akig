import sympy as sp
from numpy import array, zeros, float64, vstack

from geo_adjust.gm import AdjustHelper
from geo_utils.common.datastructure import set_values
from geo_utils.simulation.data import simulate_random_matrix

gk = array([[7.68614800e+04, 5.15634790e+06, 4.34030000e+02],
            [6.40750400e+04, 5.15428778e+06, 4.82920000e+02],
            [7.26661800e+04, 5.15536113e+06, 4.48300000e+02],
            [6.42108800e+04, 5.15364684e+06, 5.49090000e+02],
            [7.27202900e+04, 5.16040908e+06, 5.15580000e+02],
            [6.50605300e+04, 5.15564602e+06, 4.65710000e+02],
            [6.58192900e+04, 5.15708130e+06, 6.00690000e+02],
            [7.44009100e+04, 5.15457319e+06, 4.65790000e+02],
            [6.95611200e+04, 5.15814276e+06, 5.47260000e+02]])


def point_obs():
    pts = zeros((9,), dtype=[('pt', 'U10'), ('y', float64), ('x', float64), ('fix', bool)])
    pts['pt'] = ['0{}'.format(x) for x in range(1, 10)]

    obs = zeros((14,), dtype=[('pt', 'U10'), ('y', float64), ('x', float64)])
    obs['pt'] = ['0{}'.format(x) for x in range(1, 10)] + ['01', '02', '03', '04', '05']

    set_values(pts, ['y', 'x'], gk[:, :2])
    set_values(obs, ['y', 'x'], vstack((simulate_random_matrix(gk[:, :2], 0.03),
                                        simulate_random_matrix(gk[:5, :2], 0.03))))

    # define the point observation model
    y, x = sp.symbols('y x')
    phi = sp.Matrix([[y], [x]])

    solver = AdjustHelper.add_block(pts, obs, phi, (y, x), ['pt'], col_name='pt',
                                    col_param=['y', 'x'], col_obs=['y', 'x'],       # where to get data from
                                    template_param='y_{0} x_{0}', template_obs='yo_{0} xo_{0}', col_obs_name='pt')

    result = solver.solve()

    result.model.print_info()
    result.model.print_summary()

    result.parameters.print()
    result.observations.print()


if __name__ == '__main__':
    # simple_gm()
    # gh()
    point_obs()
