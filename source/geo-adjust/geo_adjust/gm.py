import sys
from collections import OrderedDict, Counter
from time import perf_counter

import sympy as sp
import numpy as np

from scipy.stats import chi2
from numpy.linalg import inv, det, cond, matrix_rank
from numpy.matlib import repmat
from scipy.linalg import block_diag, svd

from geo_adjust.result import GMResult
from geo_utils.stats.cov import compute_corr_coeff


class GMAdjust:
    def __init__(self, model_config, **kwargs):
        # store model config
        self.model = model_config
        self.model.set_solver(self)

        # supplied observations
        self._l_obs = None
        # current estimate of observations
        self._l = None
        # actual v (
        self._v = None
        # constants
        self._const = None

        self.converged = False
        self._iterations = 0

        # intial guess of parameters
        self._x0 = None
        # current estimate of parameters
        self._x = None
        self._dx = None  # change of parameters

        # computation options
        self._options = {
            'max_iterations': 30,  # maximum number of iterations before aborting
            'epsilon': 1e-12,  # convergence criterium (if max(dx) < epsilon then convergence is reached
            'alpha': 0.1,  # Irrtumswahrscheinlichkeit
            'globaltest': 'eq',  # could be 'eq' for 2-sided test or 'leq' for less than or equal (one sided)
            'journal': True,  # log the history of parameters and observations
            'verbose': True,  # print progress while iterating
        }

        # in case of kwargs treat them as options
        if kwargs:
            self.set_options(**kwargs)

        # covariance apriori
        self._var0_prio = None
        self._Sigma_ll = None
        self._Q_ll = None

        # posterior stochastic
        self._var0_post = None
        self._vtPv = None
        self._Q_xx = None
        self._R_xx = None  # Correlation Coefficient Matrix of Parameters
        self._Q_llp = None  # Cofactors of measurements aposteriori
        self._Q_vv = None
        self._R = None  # Redundancy Matrix
        self._H = None  # Hat-Matrix or Data Resolution Matrix

        self._A = None
        self._P = None

        self._Sigma_xx = None
        self._Sigma_llp = None
        self._Sigma_vv = None

        # timing information
        self._timing = {'start': perf_counter()}

        # logging/journal data
        self._x_journal = None
        self._l_journal = None

    def set_options(self, **kwargs):
        for k in self._options:
            if k in kwargs:
                self._options[k] = kwargs[k]
                kwargs.pop(k)
        if kwargs:
            raise ValueError("unknown configuration options {}.".format(kwargs))

    def _recompute_matrices(self):
        dl, A = [], []

        for b in self.model._blocks:
            dlb, Ab = b.recompute_matrices()
            dl += [dlb]
            A += [Ab]

        A = np.vstack(A)
        dl = np.vstack(dl)

        return dl, A

    def set_stochastic_model(self, sigma_ll=None, var0=1., name='default'):

        if name == 'default':
            self.model._blocks[0].set_stochastic_model(sigma_ll, var0)
        else:
            found = False
            for b in self.model._blocks:
                if b.name == name:
                    b.set_stochastic_model(sigma_ll, var0)
                    found = True
            if not found:
                raise ValueError("could not set stochastic model for block '{}', no such block.".format(name))

        # update overall Q_ll and Sigma_ll
        self._Q_ll = block_diag(*[b._Q_ll for b in self.model._blocks])
        self._Sigma_ll = block_diag(*[b._Sigma_ll for b in self.model._blocks])

    def set_initial_params(self, x):
        """
        Set x_0 (starting point for iteration)

        :param x: initial guess for parameters
        :return: None
        """
        if len(x) == self.model._u:
            self._x = np.array(x)
            self._x0 = np.array(x)
        else:
            raise ValueError("wrong dimension for initial parameters. (should be {0:d})".format(self.model._u))

    @property
    def _obs_row_names(self):
        names = []
        for b in self.model._blocks:
            if b._obs_row_names is not None:
                names += list(b._obs_row_names)
        return names

    def add_data(self, data, sigma_ll=None, var0=1., name='default', row_names=None):
        if name == 'default':
            self.model._blocks[0].add_data(data, sigma_ll, var0, row_names)
        else:
            found = False
            for b in self.model._blocks:
                if b.name == name:
                    b.add_data(data, sigma_ll, var0, row_names)
                    found = True
            if not found:
                raise ValueError("could not set data for block '{}', no such block.".format(name))

        # update global obs
        self._l_obs = np.hstack([b._l_obs.flatten() for b in self.model._blocks])

        self.set_stochastic_model(sigma_ll, var0, name)
        self._r = sum([b._r for b in self.model._blocks])
        self._n = sum([b._n for b in self.model._blocks])

    def add_constants(self, const):
        if self.model._sym_c is None:
            raise RuntimeError("no constants contained in model.")

        const = np.atleast_2d(const)
        if const.shape == (self._r, len(self.model._sym_c)):
            self._const = const
        elif const.shape == (1, len(self.model._sym_c)):
            self._const = repmat(const, self._r, 1)
        elif const.shape == (len(self.model._sym_c), self._r):
            self._const = const.T
        elif const.shape == (len(self.model._sym_c), 1):
            self._const = repmat(const.T, self._r, 1)
        elif np.isscalar(const) and len(self.model._sym_c) == 1:
            self._const = np.array([const] * self._r)
        else:
            raise ValueError('supplied dimensions of constants not understood.')

        self._t = self._const.size

        # update derivatives
        self.model._compute_derivatives()

    def solve(self):
        if self._x is None:
            raise ValueError('initial parameters missing, call set_initial_params() first.')

        self.converged = False
        self._iterations = 0

        self._timing['preparation_end'] = perf_counter()
        self._timing['iterations'] = OrderedDict()

        if self._options['verbose']:
            self._print_iteration_header()

        for i in range(self._options['max_iterations']):
            self._iterations += 1
            t_it_start = perf_counter()
            dli, Ai = self._recompute_matrices()

            t_it_recomp = perf_counter()
            vi, Ni, Pi = self._iterate(dli, Ai)
            t_it_finish = perf_counter()

            self._timing['iterations'][str(self._iterations)] = {'t_start': t_it_start,
                                                                 't_recomp': t_it_recomp,
                                                                 't_finish': t_it_finish,
                                                                 'recomp_dur': t_it_recomp - t_it_start,
                                                                 'solve_dur': t_it_finish - t_it_recomp,
                                                                 'iteration_dur': t_it_finish - t_it_start}

            if np.max(np.abs(self._dx)) < self._options['epsilon']:
                self.converged = True
                break

        self._A = Ai
        self._v = vi
        self._P = Pi

        self._compute_varcovar(Ai, Ni, Pi)

        self._timing['finish'] = perf_counter()
        # in fact this is not needed, but there are differences of around 1e-16 between v and l - l_obs
        # self._v = self._l - self._l_obs

        if self._options['verbose']:
            self._print_iteration_footer()

        def _reduce_spMat(matlist):
            result = matlist[0]
            for i in range(1, len(matlist)):
                if matlist[i] is None:
                    return None
                result = result.col_join(matlist[i])
            return result

        self._sym_l = sp.Matrix([b._sym_l for b in self.model._blocks])
        self._sym_phi = _reduce_spMat([b._sym_phi for b in self.model._blocks])
        self._sym_A = _reduce_spMat([b._sym_A for b in self.model._blocks])

        # hauptprobe & globaltest is done within GMResult()
        result = GMResult(self)
        return result

    def _obs_update(self, v):
        c = 0
        for b in self.model._blocks:
            b._v = v[c:c + b._l_obs.size]
            b._v = b._v.reshape(b._l_obs.shape)
            b._l = b._l_obs + b._v
            c += b._l_obs.size

    def _iterate(self, dl, A):
        P = inv(self._Q_ll)

        def check_inversion(N):
            if cond(N) < sys.float_info.epsilon or abs(det(N)) < sys.float_info.epsilon:
                raise ValueError("matrix 'N' singular or bad conditioned (det: {:.3f} | cond: {:.3f})".format(cond(N),
                                                                                                              abs(det(
                                                                                                                  N))))

        N = A.T @ P @ A
        # check N condition or singularity
        check_inversion(N)

        # parameters
        self._dx = np.linalg.solve(N, A.T @ P @ dl)
        # residuals
        v = A @ self._dx - dl

        self._v = v.flatten()
        self._l = self._l_obs + self._v

        # compute variance aposteriori
        self._f = (self._n - self.model._u)
        self._vtPv = (v.T @ P @ v).squeeze()
        self._var0_post = self._vtPv / self._f

        # update observations in blocks
        self._obs_update(v)

        if self._options['journal']:
            self._l_journal[self._iterations - 1, :self._n] = self._l.flatten()
            self._l_journal[self._iterations - 1, self._n:] = self._v.flatten()
            self._x_journal[self._iterations - 1, :self.model._u] = self._x.flatten() + self._dx.flatten()
            self._x_journal[self._iterations - 1, self.model._u:] = self._dx.flatten()

        if self._options['verbose']:
            self._print_iteration()

        # update parameters
        self._x += self._dx.flatten()

        return v, N, P

    def _hauptprobe(self):
        w = []
        for b in self.model._blocks:
            w += [b._hauptprobe()]

        w = np.vstack(w)
        if np.max(np.abs(w)) < self._options['epsilon']:
            return True, w.flat[np.abs(w).argmax()]
        else:
            return False, w.flat[np.abs(w).argmax()]

    def _compute_varcovar(self, A, N, P):
        # accuracy of parameters
        self._Q_xx = inv(N)[:self.model._u, :self.model._u]
        # correlations of parameters
        self._R_xx = compute_corr_coeff(self._Q_xx)

        # accuracy of observations
        self._Q_llp = A @ inv(N) @ A.T
        self._Q_vv = self._Q_ll - self._Q_llp

        self._R = self._Q_vv @ inv(self._Sigma_ll)

        self._Sigma_xx = self._Q_xx * self._var0_post
        self._Sigma_vv = self._Q_vv * self._var0_post
        self._Sigma_llp = self._Q_llp * self._var0_post

        # data resolution or hat matrix
        self._H = A @ inv(A.T @ inv(self._Sigma_ll) @ A) @ A.T @ inv(self._Sigma_ll)

        # U, s, V = svd(A, full_matrices=False)
        # V, S = V.T, np.diag(s)
        # r = matrix_rank(A)
        # Ur, Sr, Vr = U[:, :r], S[:r, :r], V[:, :r]
        #
        # # verify eq. 2.41 (Vennebusch)
        # assert np.allclose(A, U @ S @ V.T)
        #
        # A_pseudo_inv = Vr @ inv(Sr) @ Ur.T
        # # verify eq. 2.48
        # Ainv = inv(A.T @ A) @ A.T
        # assert np.allclose(Ainv, A_pseudo_inv)
        #
        # # H2 = Ur @ Ur.T
        # # assert np.allclose(self._H, H2)
        #
        # assert int(round(np.trace(self._H))) == int(round(matrix_rank(self._H))) == int(self._u)

        if len(self.model._blocks) == 1:
            self._v = self._v.reshape((self.model._blocks[0]._r, -1))
            self._l = self._l.reshape((self.model._blocks[0]._r, -1))

        # TODO: update Q_llp, Q_vv and Sigmas in blocks

    def _globaltest(self, test=None):
        """
        Execute Globaltest (ratio covariance apriori and a posteriori)
        Shows the correctness of the stochastic model
        if test = 'eq' H0: s02 = sigma02,
        if test = 'leq' H0: s02 <= sigma02
        return True if H0 can be accepted, False if H0 must be refused
        :return: accept_H0, testgroesse, critical value
        """
        # Gewichtsreziprokenprobe nach Ansermet
        assert np.round(np.trace(self._Q_vv @ inv(self._Q_ll))) == self._f

        T = self._var0_post / self._var0_prio
        F = (chi2.ppf(self._options['alpha'] / 2, self._f) / self._f,
             chi2.ppf(1 - self._options['alpha'] / 2, self._f) / self._f,
             chi2.ppf(1 - self._options['alpha'], self._f) / self._f)

        if test is None:
            test = self._options['globaltest']

        if test == 'eq':
            return not (T < F[0] or T > F[1]), T, F[:2]
        elif test == 'leq':
            return not T > F[2], T, F[2]

    @staticmethod
    def _print_iteration_header():
        print('\n======================================================')
        print(' Iterating')
        print('======================================================')
        print('')
        print('  iter \t   vtPv  \t  max(w) \t  max(x)')

    def _print_iteration(self):
        print('  {:3d} \t{:9.2e} \t{:9.2e} \t{:9.2e}'.format(self._iterations, self._vtPv,
                                                             self._hauptprobe()[1],
                                                             self._dx.flat[np.abs(self._dx).argmax()]))

    @staticmethod
    def _print_iteration_footer():
        print('\n======================================================\n')


class AdjustHelper:
    def __init__(self):
        # parameters
        self.params = []
        self.params_template = None

        # constants
        self.consts = None
        self.consts_template = None

        # measurements
        self.obs = []
        self.obs_template = None

    @staticmethod
    def _create_syms(template, names):

        if type(names[0]) in (list, tuple):
            if type(sp.symbols(template.format(*names[0]))) is sp.Symbol:
                return np.array([list((sp.symbols(template.format(*x)),)) for x in names])
            else:
                return np.array([list(sp.symbols(template.format(*x))) for x in names])
        else:
            if type(sp.symbols(template.format(names[0]))) is sp.Symbol:
                return np.array([list((sp.symbols(template.format(x)),)) for x in names])
            else:
                return np.array([list(sp.symbols(template.format(x))) for x in names])

    @staticmethod
    def _get_data(data, cols, flatten=False, filter_col=None, filter_val=None):
        """ convert columns 'cols' of recarray 'params' to numpy array
              - optionally flatten and return as list
              - optionally select rows by condition 'filter_col' == 'filter_val'"""

        if filter_col is not None and filter_val is not None:
            data = data[data[filter_col] == filter_val]

        data = data[cols].copy().view(np.float64).reshape(data[cols].shape + (-1,))
        if flatten:
            return data.flatten().tolist()
        else:
            return data

    def create_params(self, template, names):
        """ create symbols for parameters
            e.g. heights for points 01, 02, 03
            create_params('h_{0}', ['01', '02', '03'])
            e.g. 2d coordinates for points 123, 140, 200
            create_params('y_{0} x_{0}', ['123', '140', '200'])
        """
        self.params_template = template
        self.params = self._create_syms(template, names)
        return self.params

    def create_consts(self, template, names):
        """ create symbols for parameters
            e.g. heights for points 01, 02, 03
            create_params('h_{0}', ['01', '02', '03'])
            e.g. 2d coordinates for points 123, 140, 200
            create_params('y_{0} x_{0}', ['123', '140', '200'])
        """
        self.consts_template = template
        self.consts = self._create_syms(template, names)
        return self.consts

    def _unify_obs(self):
        o = self.obs.copy().flatten()
        duplicates = (Counter(o) - Counter(set(o))).keys()
        for x in duplicates:
            nums = len(self.obs[self.obs == x])
            replacements = [sp.symbols('{}^{}'.format(x.name, i)) for i in range(1, nums + 1)]
            np.place(self.obs, self.obs == x, replacements)

    def create_obs(self, template, names):
        self.obs_template = template
        self.obs = self._create_syms(template, names)
        # handle duplicate observations
        self._unify_obs()
        return self.obs

    def create_model(self, base, parts, part_finder):
        if self.obs.shape[0] != len(part_finder):
            raise ValueError("parameter 'part_finder' must be of same length as 'self.obs'.")

        # if len(parts) != len(part_finder[0]):
        #     raise ValueError("parameter 'part_finder' must contain elements of same length as 'parts'")

        model = [None] * len(self.obs)
        if self.consts is not None:
            pc_sym = np.vstack((self.params, self.consts))
        else:
            pc_sym = self.params
        pc_lookup = {}
        for p in pc_sym.flatten():
            pc_lookup[p.name] = p

        for j, p in enumerate(part_finder):
            m = []
            for i, q in enumerate(parts):
                if i < len(p):
                    name = self.params_template.format(p[i])
                else:
                    name = self.params_template.format(p[0])
                if ' ' in name:
                    names = name.split()
                    if names[i] in pc_lookup:
                        m += [(q, pc_lookup[names[i]])]
                    else:
                        raise NotImplementedError("multidimensional constants might not be found.")
                else:
                    if name in pc_lookup:
                        m += [(q, pc_lookup[name])]
                    else:
                        name = self.consts_template.format(p[i])
                        if name in pc_lookup:
                            m += [(q, pc_lookup[name])]
                        else:
                            raise ValueError("could not find variable for {}".format(p[i]))

            model[j] = base.subs(m)

        result = model[0]
        for i in range(1, len(model)):
            result = result.col_join(model[i])
        return result

    @classmethod
    def add_block(cls, params, obs, phi, parts, idx, col_const='fix', col_name='name',
                  col_param='h', col_obs='dh', col_obs_name='name', solver=None,
                  template_const='{}', template_param='{}', template_obs='{}',
                  extra_params=None, extra_params_initial=None, name='default',
                  phix=None, Ax=None):
        helper = AdjustHelper()
        # timing (for debugging)
        _timing = {}

        # create symbols for Anschlusspunkte (constants)
        const_sym = None
        _timing['block_create_components_start'] = perf_counter()
        if len(params[col_name][params[col_const] == True]):
            const_sym = helper.create_consts(template_const, params[col_name][params[col_const] == True])
        # create symbols for Neupunkte (parameters)
        param_sym = helper.create_params(template_param, params[col_name][params[col_const] == False])

        obs_sym = helper.create_obs(template_obs, obs[col_obs_name].tolist())
        _timing['block_create_components_end'] = perf_counter()

        phi_sym = helper.create_model(phi, parts, obs[idx].tolist())
        _timing['block_create_model_end'] = perf_counter()

        # create new Solver if necessary
        if solver is None:
            solver = GMAdjust(verbose=False, journal=False)

        p = tuple(param_sym.flatten())
        if extra_params:
            p += tuple(extra_params)
        consts = tuple(const_sym.flatten()) if const_sym is not None else None
        if phix is not None and Ax is not None:
            solver.add_model(phi_sym, p, tuple(obs_sym.flatten()),
                             phix, Ax,
                             c=consts, name=name)
        else:
            solver.add_model_autodiff(phi_sym, p, tuple(obs_sym.flatten()),
                                      c=consts, name=name)
        _timing['block_add_model_autodiff_end'] = perf_counter()

        p0 = AdjustHelper._get_data(params, col_param,
                                   filter_col=col_const, filter_val=False,
                                   flatten=True)
        if extra_params:
            p0 += list(extra_params_initial)

        solver.set_initial_params(p0)
        # solver.add_data(obs[col_obs])
        solver.add_data(AdjustHelper._get_data(obs, col_obs, flatten=True), name=name)
        # solver.add_constants(params[col_param][params[col_const] == True].tolist())
        if const_sym is not None:
            solver.add_constants(AdjustHelper._get_data(params, col_param,
                                                        filter_col=col_const, filter_val=True,
                                                        flatten=True))
        _timing['block_add_data_end'] = perf_counter()

        # print(' B L O C K ({}):'.format(name))
        # print(' - Component Creation:       {:9.3f} [ms]'.format(
        #     (_timing['block_create_components_end'] - _timing['block_create_components_start']) * 1e3))
        # print(' - Model Creation:           {:9.3f} [ms]'.format(
        #     (_timing['block_create_model_end'] - _timing['block_create_components_end']) * 1e3))
        # print(' - Add Model Autodiff:       {:9.3f} [ms]'.format(
        #     (_timing['block_add_model_autodiff_end'] - _timing['block_create_model_end']) * 1e3))
        # print(' - Add Data                  {:9.3f} [ms]'.format(
        #     (_timing['block_add_data_end'] - _timing['block_add_model_autodiff_end']) * 1e3))

        return solver
