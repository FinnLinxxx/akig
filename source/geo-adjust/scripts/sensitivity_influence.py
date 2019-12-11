from os import listdir
from os.path import isfile, join, basename

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def gon2rad(arr):
    arr[:, 3] = arr[:, 3]/200*np.pi
    return arr


if __name__ == '__main__':
    path = 'examples/data'

    # load H and Rs
    HRs = pd.read_pickle(join(path, 'HRs.pk'))

    # read simulation
    full = pd.read_pickle(join(path, 'params_full.pk'))
    full = gon2rad(full)
    full_var = np.var(full, axis=0)

    # observations names
    obs_names = ['y{0}_u x{0}_u'.format(i) for i in range(1, 6)]
    obs_names += ['y{0}_t x{0}_t'.format(i) for i in range(1, 6)]
    obs_names = ' '.join(obs_names).split()
    pt_names = ['ur_{0:02d}'.format(i) for i in range(1,6)] + ['tp_{0:02d}'.format(i) for i in range(1,6)]

    # read simulation results partial
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    vars = []
    parts = {}

    for f in sorted(files):
        if basename(f) != 'HRs.pk' and basename(f) != 'params_full.pk' and basename(f).startswith('param'):
            part = pd.read_pickle(f)
            part = gon2rad(part)
            parts[basename(f)] = part
            part_var = np.var(part, axis=0)
            vars += [part_var]

    cols = ['dy', 'dx', 'm', 'alpha']

    factor = 0.9

    df = pd.DataFrame(np.vstack(vars), index=obs_names, columns=cols)
    sensitivity = pd.DataFrame((1 - (df.values / full_var)).T / factor * 100,
                               index=cols, columns=obs_names)

    distribution = pd.DataFrame(sensitivity.values / np.sum(sensitivity.values, 0),
                                index=cols, columns=obs_names)

    contribution = pd.DataFrame(sensitivity.values / np.sum(sensitivity.values, 1)[:, None],
                                index=cols, columns=obs_names)

    np.testing.assert_allclose(np.sum(distribution, 0), 1)
    np.testing.assert_allclose(np.sum(contribution, 1), 1)

    sensitivity.to_pickle(join('examples/data', 'sensitivity_sim.pk'))

    print('OK')
