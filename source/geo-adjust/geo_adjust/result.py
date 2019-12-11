from collections import OrderedDict

from numpy import diag, sqrt, block, atleast_2d, any, abs, quantile, around, arange, fill_diagonal
from sympy import pretty, latex
from pandas import DataFrame

from tabulate import tabulate
from colorama import init, Fore, Style, Back

# true false symbols: ✗🗶 and ✓✔ or ☐ ☑ ☒ ⮽ 🗵 🗹
from geo_adjust.utils import lookahead
from geo_adjust.utils import format_figure
from geo_adjust.plot import heatmap, annotate_heatmap


class AdjustmentModel:
    def __init__(self, solver):
        # init colorama
        init(autoreset=True)

        self.convergence = solver.converged
        self.epsilon = solver._options['epsilon']
        self.num_iterations = solver._iterations
        # hauptprobe
        self.check, self.max_w = solver._hauptprobe()

        # timing information
        self.timing = solver._timing

        # model defintion
        self.parameters = {'unicode': [pretty(x) for x in solver.model._sym_x],
                           'latex': [latex(x) for x in solver.model._sym_x]}
        self.observations = {'unicode': [pretty(x) for x in solver.model._sym_l],
                             'latex': [latex(x) for x in solver.model._sym_l]}

        self.auto_diff = True if solver.model._sym_phi else False
        if self.auto_diff:
            self.phi = {'unicode': pretty(solver.model._sym_phi), 'latex': latex(solver.model._sym_phi)}
            self.A = {'unicode': pretty(solver.model._sym_A), 'latex': latex(solver.model._sym_A)}

        # constants
        if solver.model._sym_c is not None:
            self.constants = {'unicode': [pretty(x) for x in solver.model._sym_c],
                              'latex': [latex(x) for x in solver.model._sym_c]}
            self.constants_values = solver._const
        else:
            self.constants = None
            self.constants_values = None

        # several numbers
        self.num_parameters = solver.model._u
        self.num_observations = solver._n
        self.num_constants = solver._t
        self.dof = solver._f  # degree of freedom (Gesamtredundanz)

        # goodness of fit
        self.sse = solver._v.flatten() @ solver._v.flatten()
        self.mse = self.sse / len(solver._v.flatten())
        self.rmse = sqrt(self.mse)

        # weighted ones
        self.sse2 = solver._vtPv
        self.mse2 = self.sse2 / self.dof
        self.rmse2 = self.sse2 / self.dof

        self.var_prio_post = (solver._var0_prio, solver._var0_post)
        self.test_result, self.test, self.bound = solver._globaltest()
        self.test_h0 = solver._options['globaltest']
        self.test_alpha = solver._options['alpha']

    def print_timing(self):
        print('\n======================================================')
        print(' Timing Result ')
        print('======================================================')

        print(' - Adjustment Preparation: \t  {:9.3f} [ms]'.format(
            (self.timing['preparation_end'] - self.timing['start']) * 1e3))
        print(' - Iterative Solution:     \t  {:9.3f} [ms]'.format(
            (self.timing['finish'] - self.timing['preparation_end']) * 1e3))
        print('                           \t ---------------')
        print('                    Total: \t  {:9.3f} [ms]'.format(
            (self.timing['finish'] - self.timing['start']) * 1e3))

        print('')
        print('  iter \t recomp [ms] \t  solve [ms] \t  total [ms]')
        for k, v in self.timing['iterations'].items():
            print('  {:3d} \t {:8.3f} \t {:11.3f} \t {:11.3f}'.format(int(k), v['recomp_dur'] * 1e3,
                                                                      v['solve_dur'] * 1e3,
                                                                      v['iteration_dur'] * 1e3))
        print('\n======================================================\n')

    def print_summary(self):
        print('\n======================================================')
        print(' Result Summary ')
        print('======================================================')
        print(' - Convergence: \t\t\t\t  {:1s}'.format('[ OK]' if self.convergence else '[NOK]'))
        print('    ├ # Iterations: \t\t\t  {:4d}'.format(self.num_iterations))
        print('    └ Criterium: \t\t\t\t\t  {:7.3e}'.format(self.epsilon))
        print(' - Hauptprobe: \t\t\t\t  {:1s}'.format('[ OK]' if self.check else '[NOK]'))
        print('    └ Max Eps: \t\t\t\t\t  {:7.3e}'.format(self.max_w))
        print(' - Globaltest: \t\t\t\t  {:1s}'.format('[ OK]' if self.test_result else '[NOK]'))
        print('    ├ Test: \t\t\t\t\t\t  {:3s}'.format(self.test_h0))
        print('    ├ Var Prio: \t\t\t\t\t {:6.3f}'.format(self.var_prio_post[0]))
        print('    ├ Var Post: \t\t\t\t\t {:6.3f}'.format(self.var_prio_post[1]))
        print('    ├ Ratio:    \t\t\t\t\t {:6.3f}'.format(self.test))
        print('    ├ Alpha: \t\t\t\t\t\t  {:.3f}'.format(self.test_alpha))
        if self.test_h0 == 'eq':
            print(
                '    └ Testquantity: \t\t\t\t  {1:.3f} < {0:.3f} < {2:.3f}'.format(self.test, self.bound[0],
                                                                                   self.bound[1]))
        elif self.test_h0 == 'leq':
            print('    └ Testquantity: \t\t\t\t  {1:.3f} ⩽ {0:.3f}'.format(self.test, self.bound))
        else:
            raise NotImplementedError()

        # Residuals
        print(' - Residuals:')
        print('    ├ SSE (vTv):   {:18.3f}'.format(self.sse))
        print('    ├ MSE:         {:18.3f}'.format(self.mse))
        print('    ├ RMSE:        {:18.3f}'.format(self.rmse))
        print('    ├ WSSE (vTPv): {:18.3f}'.format(self.sse2))
        print('    ├ WMSE:        {:18.3f}'.format(self.mse2))
        print('    └ WRMSE:       {:18.3f}'.format(self.rmse2))

        print('\n======================================================\n')


class GHModel(AdjustmentModel):
    def __init__(self, solver):
        super().__init__(solver)

        self.short_name = 'GH+C' if solver.model._constraint else 'GH'
        self.name = 'Gauss-Helmert-Model with Constraints' if solver.model._constraint else 'Gauss-Helmert-Model'

        if self.auto_diff:
            self.B = {'unicode': pretty(solver.model._sym_B), 'latex': latex(solver.model._sym_B)}

        # constraints definition
        self.constraint = True if solver.model._constraint else False
        self.constraint_auto_diff = True if solver.model._sym_gamma else False
        if self.constraint_auto_diff:
            self.gamma = {'unicode': pretty(solver.model._sym_gamma), 'latex': latex(solver.model._sym_gamma)}
            self.C = {'unicode': pretty(solver.model._sym_C), 'latex': latex(solver.model._sym_C)}

        # several numbers
        self.num_equations = solver._r
        self.num_constraints = solver.model._c

    def print_info(self, print_matrices=False):
        print('\n======================================================')
        print(' Adjustment Model ')
        print('======================================================')

        print(' - Adjustment Model: \t\t  {:4s}'.format(self.short_name))
        print('              ({:s})'.format(self.name))
        print(' - Observations ({:2d}): \t  {:s}'.format(int(self.num_observations / self.num_equations),
                                                         ', '.join(self.observations['unicode'])))
        print('    └ Total #: {:4d}\n'.format(self.num_observations))
        print(' - Parameters   ({:2d}): \t  {:s}'.format(self.num_parameters,
                                                         ', '.join(self.parameters['unicode'])))
        if self.constants:
            print(' - Constants    ({:2d}): \t  {:s}'.format(len(self.constants['unicode']),
                                                             ', '.join(self.constants['unicode'])))
            print('    └ Total #: {:4d}\n'.format(self.num_constants))

        print('')
        print(' - Redundancy: \t\t\t\t  {:4d}'.format(self.dof))
        print('    ├ # of Parameters: \t\t  {:4d}'.format(self.num_parameters))
        print('    ├ # of Observations: \t  {:4d}'.format(self.num_observations))
        if self.constraint:
            print('    ├ # of Constraints: \t  {:4d}'.format(self.num_constraints))
        print('    └ # of Equations: \t\t  {:4d}'.format(self.num_equations))

        print('\n------------------------------------------------------')
        print(' - Model:')
        print('    ├ Auto Differentiation:\t  {:1s}'.format('[ OK]' if self.auto_diff else '[NOK]'))
        print('    └ Phi(x,l) = 0:')
        for i in self.phi['unicode'].splitlines():
            print('        {}'.format(i))
        print('')

        if print_matrices:
            print('    └ Designmatrix A:')
            for i in self.A['unicode'].splitlines():
                print('        {}'.format(i))
            print('\n    └ Observations Jacobian B:')
            for i in self.B['unicode'].splitlines():
                print('        {}'.format(i))
            print('')

        if self.constraint:
            print(' - Constraint:')
            print(
                '    ├ Auto Differentiation: \t  {:1s}'.format('[ OK]' if self.constraint_auto_diff else '[NOK]'))
            print('    └ Gamma(x) = 0:')
            for i in self.gamma['unicode'].splitlines():
                print('        {}'.format(i))

            if print_matrices:
                print('')
                print('    └ Constraint Jacobian C:')
                for i in self.C['unicode'].splitlines():
                    print('        {}'.format(i))

        # print('\n------------------------------------------------------')
        print('\n======================================================\n')


class GMModel(AdjustmentModel):
    def __init__(self, solver):
        super().__init__(solver)

        self.short_name = 'GM'
        self.name = 'Gauss-Markov-Model'

    def print_info(self, print_matrices=False):
        print('\n======================================================')
        print(' Adjustment Model ')
        print('======================================================')

        print(' - Adjustment Model: \t\t  {:4s}'.format(self.short_name))
        print('              ({:s})'.format(self.name))
        print(' - Observations ({:2d}): \t  {:s}'.format(int(self.num_observations),
                                                         ', '.join(self.observations['unicode'])))
        print(' - Parameters   ({:2d}): \t  {:s}'.format(self.num_parameters,
                                                         ', '.join(self.parameters['unicode'])))
        if self.constants:
            print(' - Constants    ({:2d}): \t  {:s}'.format(int(self.num_constants),
                                                             ', '.join(self.constants['unicode'])))

        print('')
        print(' - Redundancy: \t\t\t\t  {:4d}'.format(self.dof))
        print('    ├ # of Parameters: \t\t  {:4d}'.format(self.num_parameters))
        print('    ├ # of Observations: \t  {:4d}'.format(self.num_observations))

        print('\n------------------------------------------------------')
        print(' - Model:')
        print('    ├ Auto Differentiation:\t  {:1s}'.format('[ OK]' if self.auto_diff else '[NOK]'))
        print('    └ Phi(l) = x:')
        for i in self.phi['unicode'].splitlines():
            print('        {}'.format(i))
        print('')

        if print_matrices:
            print('    └ Designmatrix A:')
            for i in self.A['unicode'].splitlines():
                print('        {}'.format(i))
            print('')

        # print('\n------------------------------------------------------')
        print('\n======================================================\n')


class AdjustmentParameters:
    def __init__(self, solver):
        self.names = {'unicode': [pretty(x) for x in solver.model._sym_x],
                      'latex': [latex(x) for x in solver.model._sym_x]}

        self.syms = solver.model._sym_x
        self.params_config = OrderedDict()

        # values of parameters
        self.initial_values = solver._x0
        self.values = solver._x
        self.vcm = solver._Sigma_xx
        self.var = diag(self.vcm).flatten()
        self.std = sqrt(self.var)
        self.corr = solver._R_xx

        self.vcm = solver._Sigma_xx

        self.journal = solver._x_journal

    def set_params_config(self, params, units, names, cxs=None, formats=None):
        for i, p in enumerate(params):
            idx = self.syms.index(p)
            if cxs is not None:
                self.params_config[names[i]] = {'idx': idx, 'unit': units[i], 'p': p, 'cx': cxs[i]}
            else:
                self.params_config[names[i]] = {'idx': idx, 'unit': units[i], 'p': p, 'cx': None}

            if formats is not None:
                self.params_config[names[i]]['fmt'] = formats[i]

    def print(self, print_correlation=True):
        print('\n======================================================')
        print(' Paramters')
        print('======================================================')

        print(' - Parameter Estimates ± σ:')
        if self.params_config:
            for (k, p), has_more in lookahead(self.params_config.items()):

                if p['cx'] is not None:
                    if type(p['cx']) is tuple:
                        converted_v = p['cx'][0](self.values[p['idx']])
                        converted_std = p['cx'][1](self.std[p['idx']])
                    else:
                        converted_v = p['cx'](self.values[p['idx']])
                        converted_std = p['cx'](self.std[p['idx']])
                else:
                    converted_v = self.values[p['idx']]
                    converted_std = self.std[p['idx']]

                # format
                ft = p['fmt'] if 'fmt' in p else '.3f'

                if has_more:
                    ft_string = '    ├ {{:>4s}}: {{:12{0:}}} ± {{:8{0:}}}   [{{:s}}]'.format(ft)
                else:
                    ft_string = '    └ {{:>4s}}: {{:12{0:}}} ± {{:8{0:}}}   [{{:s}}]'.format(ft)

                print(ft_string.format(self.names['unicode'][p['idx']],
                                       converted_v, converted_std,
                                       p['unit']))

        else:
            for (i, p), has_more in lookahead(enumerate(self.names['unicode'])):
                if has_more:
                    print('    ├ {:>4s}: {:12.3f} ± {:8.3f}'.format(p,
                                                                    self.values[i],
                                                                    self.std[i]))
                else:
                    print('    └ {:>4s}: {:12.3f} ± {:8.3f}'.format(p,
                                                                    self.values[i],
                                                                    self.std[i]))

        # TODO confidence bounds/interval
        # TODO global accuracy measures (Eigenvalues of Sigmaxx)

        if print_correlation:
            print('- Correlation Matrix:')
            print_data = self.corr.tolist()

            for i, r in enumerate(print_data):
                for j, c in enumerate(r):
                    if c > 0.9999:
                        pre = Style.DIM
                    elif c > 0.5:
                        pre = Fore.RED + Style.BRIGHT
                    elif c > 0.1:
                        pre = Fore.LIGHTRED_EX
                    elif c < -0.5:
                        pre = Fore.BLUE + Style.BRIGHT
                    elif c < -0.1:
                        pre = Fore.LIGHTBLUE_EX
                    else:
                        pre = Style.RESET_ALL
                    c = pre + '{:.2f}'.format(c) + Style.RESET_ALL
                    r[j] = c
                print_data[i] = r

            for l in tabulate(print_data,
                              tablefmt='fancy_grid', showindex=False, floatfmt='.2f').splitlines():
                print('    {}'.format(l))

        print('\n======================================================\n')

    def plot_corr(self, fig=None, ax=None):
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.gca()

        im, cbar = heatmap(self.corr, ['${}$'.format(i) for i in self.names['latex']],
                           ['${}$'.format(i) for i in self.names['latex']], ax=ax,
                           cmap="RdBu_r", cbarlabel="corr []", vmin=-1, vmax=1)
        texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=[0.1, 0.4, 0.6],
                                 textcolors=['#9f9f9f', '#4f4f4f', 'k', 'white'], size='small')

        fig.tight_layout()
        plt.show()


class AdjustmentObservations:
    def __init__(self, solver):
        self.names = {'unicode': [pretty(x) for x in solver.model._sym_l],
                      'latex': [latex(x) for x in solver.model._sym_l]}

        self.syms = solver.model._sym_l
        self.obs_config = OrderedDict()
        self.row_names = solver._obs_row_names

        # values of observations
        self.observed_values = solver._l_obs
        self.values = solver._l
        self.v = solver._v
        self.vcm_prio = solver._Sigma_ll
        self.vcm = solver._Sigma_llp
        self.var = diag(self.vcm).reshape(self.values.shape)
        self.std = sqrt(self.var)
        self.journal = solver._l_journal
        self.vcm_v = solver._Sigma_vv
        self.var_v = diag(self.vcm_v).reshape(self.values.shape)
        self.nv = around(abs(self.v) / sqrt(self.var_v), 1)

    def set_obs_config(self, obs, units, names, cxs=None, formats=None, groups=None):
        for i, p in enumerate(obs):
            idx = self.syms.index(p)
            if cxs is not None:
                self.obs_config[names[i]] = {'idx': idx, 'unit': units[i], 'p': p, 'cx': cxs[i]}
            else:
                self.obs_config[names[i]] = {'idx': idx, 'unit': units[i], 'p': p, 'cx': None}

            if formats is not None:
                self.obs_config[names[i]]['fmt'] = formats[i]

            if groups is not None:
                self.obs_config[names[i]]['group'] = groups[i]

    def _get_values(self):
        ft, units, groups = [], [], []
        values = {'values': self.values.copy(), 'v': self.v.copy(), 'std': self.std.copy(), 'nv': self.nv.copy()}
        if self.obs_config:
            for k, p in self.obs_config.items():
                if p['cx'] is not None:
                    if type(p['cx']) is tuple:
                        values['values'][:, p['idx']] = p['cx'][0](self.values[:, p['idx']])
                        values['v'][:, p['idx']] = p['cx'][1](self.v[:, p['idx']])
                        values['std'][:, p['idx']] = p['cx'][1](self.std[:, p['idx']])
                    else:
                        values['values'][:, p['idx']] = p['cx'](self.values[:, p['idx']])
                        values['v'][:, p['idx']] = p['cx'](self.v[:, p['idx']])
                        values['std'][:, p['idx']] = p['cx'](self.std[:, p['idx']])

                if 'group' in p:
                    groups += [p['group']]
                # format
                diff_fmts = False
                if 'fmt' in p:
                    if type(p['fmt']) is tuple:
                        diff_fmts = True
                    ft += [p['fmt']]
                else:
                    ft += ['.3f']

                units += [p['unit']] if 'unit' in p else '?'
                # TODO: update units handling

            if not diff_fmts:
                # simple case (one format for all)
                ft += ft * 2 + ['.1f'] * len(ft)
            else:
                ft2, ft1 = [], []
                for f in ft:
                    ft1 += [f[0]]
                    ft2 += [f[1]]
                ft = ft1 + ft2 * 2 + ['.1f'] * len(ft)
            # ft = ft * 3
            headers = ['{0:s} [{1:s}]'.format(x, units[i]) for i, x in enumerate(self.names['unicode'])] + \
                      ['v_{:s} [{:s}]'.format(x, units[i]) for i, x in enumerate(self.names['unicode'])] + \
                      ['σ_{:s} [{:s}]'.format(x, units[i]) for i, x in enumerate(self.names['unicode'])] + \
                      ['nv_{:s} [{:s}]'.format(x, '') for i, x in enumerate(self.names['unicode'])]
        else:
            ft = '.3f'
            headers = self.names['unicode'] + \
                      ['v_{:s}'.format(x) for x in self.names['unicode']] + \
                      ['σ_{:s}'.format(x) for x in self.names['unicode']] + \
                      ['nv_{:s}'.format(x) for x in self.names['unicode']]

        return groups, values, headers, ft, units

    def _do_grouping(self, groups, data, headers, ft, units):
        unique_groups = []
        for x in groups:
            if x not in unique_groups:
                unique_groups.append(x)

        grouping = {}
        for g in unique_groups:
            grouping[g] = {'idx': [i for i, e in enumerate(groups) if e == g]}

        for k, g in grouping.items():
            g['data'] = dict(values=data['values'][:, g['idx']], v=data['v'][:, g['idx']], std=data['std'][:, g['idx']],
                             nv=data['nv'][:, g['idx']])
            g['header'] = [headers[i] for i in g['idx']] + \
                          [headers[i + len(groups)] for i in g['idx']] + \
                          [headers[i + 2 * len(groups)] for i in g['idx']] + \
                          [headers[i + 3 * len(groups)] for i in g['idx']]
            g['ft'] = [ft[i] for i in g['idx']] * 3 + ['.1f'] * 3
            g['units'] = [units[i] for i in g['idx']] * 3

        return grouping

    def dataframe(self):
        groups, data, headers, ft, units = self._get_values()

        if groups:
            dfs = {}
            grouping = self._do_grouping(groups, data, headers, ft, units)

            for k, g in grouping.items():
                g_data = atleast_2d(block([g['data']['values'], g['data']['v'],
                                           g['data']['std'], g['data']['nv']]))
                if self.row_names is not None:
                    dfs[k] = DataFrame(g_data, columns=g['header'], index=self.row_names)
                else:
                    dfs[k] = DataFrame(g_data, columns=g['header'])

            return dfs
        else:
            data = atleast_2d(block([data['values'], data['v'], data['std'], data['nv']]))
            if self.row_names is not None:
                return DataFrame(data, columns=headers, index=self.row_names)
            else:
                return DataFrame(data, columns=headers)

    def print(self):
        print('\n======================================================')
        print(' Measurements')
        print('======================================================')

        # print(' - Observed Measurements:')
        # for l in tabulate(self.observations.observed_values,
        #                   tablefmt='fancy_grid', showindex=False, floatfmt='.3f').splitlines():
        #     print('    {}'.format(l))

        print(' - Adjusted Measurements, residuals, and standard deviation (σ):')

        groups, data, headers, ft, units = self._get_values()

        # do the grouping
        if groups:
            grouping = self._do_grouping(groups, data, headers, ft, units)

            for k, g in grouping.items():
                g_data = atleast_2d(block([g['data']['values'], g['data']['v'],
                                              g['data']['std'], g['data']['nv']]))
                print('\n      {:s}'.format(k))
                print('    ------------------------------------------------------')
                print_data = g_data.tolist()

                if self.row_names is not None:
                    # add row name
                    # q1, q2 = quantile(abs(self.v[:, g['idx']]), 0.8), quantile(abs(self.v[:, g['idx']]), 0.9)
                    q1, q2 = 2., 3.
                    for k, i in enumerate(print_data):
                        if any(abs(data['nv'][k, g['idx']]) > q1):
                            if any(abs(data['nv'][k, g['idx']]) > q2):
                                i = [Fore.RED + str(self.row_names[k])] + i
                            else:
                                i = [Fore.YELLOW + str(self.row_names[k])] + i
                        else:
                            i = [str(self.row_names[k])] + i
                        print_data[k] = i

                    g['header'] = ['name'] + g['header']
                    g['ft'] = ['s'] + g['ft']

                for l in tabulate(print_data,
                                  headers=g['header'],
                                  tablefmt='fancy_grid', showindex=False, floatfmt=g['ft']).splitlines():
                    if Fore.RED in l:
                        print('    {}'.format(Fore.RED + l))
                    elif Fore.YELLOW in l:
                        print('    {}'.format(Fore.YELLOW + l))
                    else:
                        print('    {}'.format(l))
        else:
            data = atleast_2d(block([data['values'], data['v'], data['std'], data['nv']]))
            print_data = data.tolist()

            if self.row_names is not None:
                # add row name
                for k, i in enumerate(print_data):
                    i = [self.row_names[k]] + i
                    print_data[k] = i

                headers = ['name'] + headers

            for l in tabulate(print_data,
                              headers=headers,
                              tablefmt='fancy_grid', showindex=False, floatfmt=ft).splitlines():
                print('    {}'.format(l))

        print('\n======================================================\n')

    def plot(self):
        import matplotlib.pyplot as plt
        groups, data, headers, ft, units = self._get_values()

        if groups:
            grouping = self._do_grouping(groups, data, headers, ft, units)
            n = len(grouping)

            main_ax = None
            for i, (k, g) in enumerate(grouping.items()):
                if main_ax is None:
                    main_ax = plt.subplot(n, 1, i + 1)
                else:
                    plt.subplot(n, 1, i+1, sharex=main_ax)
                self._plot_residuals(g['data'], g['header'], g['units'])

            format_figure(share_x=True)
        else:
            self._plot_residuals(data, headers, units[0])
            format_figure()

    def _plot_residuals(self, data, labels, unit):
        import matplotlib.pyplot as plt

        # number of residuals to plot per row
        n = data['v'].shape[1]

        # set width of bar
        barWidth = 1 / (n + 1)

        dtx, idx = [], []
        for i in range(n):
            dtx += [data['v'][:, i]]

        # Set position of bar on X axis
        idx += [arange(data['v'].shape[0])]
        for i in range(1, data['v'].shape[1]):
            idx += [[x + barWidth for x in idx[i - 1]]]

        # zero line as thick as frame
        plt.axhline(0, color='black', ls='--', lw=plt.gca().spines['top'].get_linewidth())

        # Make the plot
        # labels = ['v_{:s}'.format(x) for x in self.names['latex']]
        for i in range(n):
            plt.bar(idx[i], dtx[i], color='C{}'.format(i), width=barWidth, edgecolor='white', label=labels[i])

        # Add xticks on the middle of the group bars
        if self.row_names:
            xlabels = self.row_names
        else:
            xlabels = ['{}'.format(i) for i in range(self.v.shape[0])]
        plt.xlabel('observation row')
        plt.ylabel('residual {}'.format(unit[0]))
        plt.xticks([r + (n / 2 - 0.5) * barWidth for r in range(self.v.shape[0])],
                   xlabels)

        # Create legend
        plt.legend()


class AdjustmentResult:
    def __init__(self, solver):
        self.parameters = AdjustmentParameters(solver)

        self.observations = AdjustmentObservations(solver)

        self.x = self.parameters.values
        self.sigma_x = self.parameters.std

        self.l = self.observations.values
        self.sigma_l = self.parameters.std


class GHResult(AdjustmentResult):
    def __init__(self, solver):
        super().__init__(solver)
        # add model information
        self.model = GHModel(solver)


class GMResult(AdjustmentResult):
    def __init__(self, solver):
        super().__init__(solver)
        # add model information
        self.model = GMModel(solver)

# parameters:
#  - value
#  - sigma
#  - confidence interval
#  - helmert und werkmeister
#  - covariance matrix
#  - Fehlerellipse
#  - Konfidenzellipse
#  - Fusspunktskurve
#  - Korrelationen
#  - confidence and prediction bounds (https://de.mathworks.com/help/curvefit/confidence-and-prediction-bounds.html)
#
#
# observations:
#  - value
#  - sigma
#  - Verbesserung
#  - normierte verbesserung
#  - Testgröße
#  - Redundanzanteil
#  - Innere Zuverlässigkeit
#  - Äußere - " -
