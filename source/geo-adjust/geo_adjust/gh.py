import sys
from collections import OrderedDict
from time import perf_counter

import numpy as np
from numpy.matlib import repmat

from scipy.stats import chi2
from numpy.linalg import inv, det, cond

from geo_adjust.result import GHResult
from geo_adjust.utils import compute_corr_coeff


class GHAdjust:
    def __init__(self, model_config, **kwargs):
        # store model config
        self.model = model_config

        # computation options
        self._options = {
            'max_iterations': 30,  # maximum number of iterations before aborting
            'epsilon': 1e-12,  # convergence criterium (if max(dx) < epsilon then convergence is reached
            'alpha': 0.1,  # Irrtumswahrscheinlichkeit
            'globaltest': 'eq',  # could be 'eq' for 2-sided test or 'leq' for less than or equal (one sided)
            'journal': False,  # log the history of parameters and observations
            'verbose': False,  # print progress while iterating
        }

        # in case of kwargs treat them as options
        if kwargs:
            self.set_options(**kwargs)

        # supplied observations
        self._l_obs = None
        # current estimate of observations
        self._l = None
        # actual v (
        self._v = None
        # constants
        self._const = None
        # observations treated as constants (idx of flattened l_obs)
        self._coidx = None
        # names for each equation block
        self._obs_row_names = None

        self.converged = False
        self._iterations = 0

        # intial guess of parameters
        self._x0 = None
        # current estimate of parameters
        self._x = None
        self._dx = None  # change of parameters

        self._n = 0  # number of total observations
        self._r = 0  # number of rows (observations)
        self._t = 0  # number of constants

        # covariance apriori
        self._var0_prio = None
        self._Sigma_ll = None
        self._Q_ll = None

        # posterior stochastic
        self._var0_post = None
        self._vtPv = None
        self._Q_xx = None
        self._R_xx = None
        self._Q_llp = None
        self._Q_vv = None

        self._Sigma_xx = None
        self._Sigma_llp = None
        self._Sigma_vv = None
        self._H = None

        self._A = None
        self._P = None
        self._B = None

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

        # initialize journal data if required
        if self._options['journal']:
            self._x_journal = np.zeros((self._options['max_iterations'], len(self.model._sym_x) * 2))

    def _recompute_matrices(self):
        w, A = [], []
        r, c = self._l_obs.shape
        B = np.zeros((r*self.model._q, r * c))
        # B = dok_matrix((r, r*c), dtype=np.float64)
        for i, l in enumerate(self._l):
            if self.model._sym_c:
                w += [self.model._model(*self._x, *l, *self._const[i, :])]
                A += [self.model._model_A(*self._x, *l, *self._const[i, :])]
                B[i*self.model._q:i*self.model._q+self.model._q, i*c:(i+1)*c] = self.model._model_B(*self._x, *l, *self._const[i, :])
            else:
                w += [self.model._model(*self._x, *l)]
                A += [self.model._model_A(*self._x, *l)]
                B[i * self.model._q:i * self.model._q + self.model._q, i * c:(i + 1) * c] = self.model._model_B(*self._x, *l)

        A = np.vstack(A)
        w = np.vstack(w) - B @ (self._l - self._l_obs).flatten()[:, None]

        return w, A, B

    def set_stochastic_model(self, sigma_ll=None, var0=1.):
        self._var0_prio = var0
        if sigma_ll is None:
            # self._Sigma_ll = dia_matrix((np.array([1] * self._n), [0]), shape=(self._n, self._n))
            self._Sigma_ll = np.eye(self._n)
        else:
            if sigma_ll.squeeze().shape == (self._n,):
                # supplied n variances
                self._Sigma_ll = np.diag(sigma_ll.squeeze())
            elif sigma_ll.shape == (self._n, self._n):
                self._Sigma_ll = sigma_ll
            else:
                raise ValueError("invalid dimension for parameter 'sigma_ll'. (should be {0:d}x{0:d} or {0:d})".format(self._n))

        self._Q_ll = 1 / self._var0_prio * self._Sigma_ll

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

    def add_data(self, data, sigma_ll=None, var0=1., row_names=None):
        data = np.atleast_2d(data)
        if data.shape[1] != self.model._m:
            raise ValueError('wrong number of observations in supplied data. (should be {0:d})'.format(self.model._m))
        # store observations
        self._l_obs = data.copy()
        # store dimensions
        self._r, c = data.shape
        # total number of observations
        self._n = self._r * c

        if row_names is not None:
            if len(row_names) != self._r:
                raise ValueError('wrong number of row_names. (should be {:d}'.format(self._r))
            self._obs_row_names = row_names

        if self._options['journal']:
            self._l_journal = np.zeros((self._options['max_iterations'], self._n * 2))

        # initialize stochastic model
        self.set_stochastic_model(sigma_ll, var0)

    def add_constants(self, const):
        if self.model._sym_c is None:
            raise RuntimeError("no constants contained in model.")

        const = np.atleast_2d(const)
        if const.shape == (self._r, len(self.model._sym_c)):
            self._const = const
        elif const.shape == (len(self.model._sym_c), self._r):
            self._const = const.T
        elif np.isscalar(const) and len(self.model._sym_c) == 1:
            self._const = np.array([const]*self._r)
        elif const.shape == (1, len(self.model._sym_c)):
            self._const = repmat(const, self._r, 1)
        else:
            raise NotImplementedError

        self._t = self._const.size

        # update derivatives
        self.model._compute_derivatives()

    def solve(self):
        if self._x is None:
            raise ValueError('initial parameters missing, call set_initial_params() first.')

        self.converged = False
        self._iterations = 0

        # starting point for real observations is data
        self._l = self._l_obs.copy()

        self._timing['preparation_end'] = perf_counter()
        self._timing['iterations'] = OrderedDict()

        if self._options['verbose']:
            self._print_iteration_header()

        for i in range(self._options['max_iterations']):
            self._iterations += 1
            t_it_start = perf_counter()
            wi, Ai, Bi = self._recompute_matrices()

            if self.model._constraint:
                wci = self.model._constraint(*self._x)
                Ci = self.model._constraint_C(*self._x)

            t_it_recomp = perf_counter()

            if self.model._constraint:
                vi, Ni, Mi = self._iterate(wi, Ai, Bi, wci, Ci)
            else:
                vi, Ni, Mi = self._iterate(wi, Ai, Bi)

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
        self._B = Bi
        self._P = inv(self._Q_ll)

        if self.model._constraint:
            self._compute_varcovar(Ai, Bi, -Ni, Mi)
        else:
            self._compute_varcovar(Ai, Bi, Ni, Mi)

        self._timing['finish'] = perf_counter()
        # in fact this is not needed, but there are differences of around 1e-16 between v and l - l_obs
        # self._v = self._l - self._l_obs

        if self._options['verbose']:
            self._print_iteration_footer()

        # hauptprobe & globaltest is done within GHResult()
        result = GHResult(self)
        return result

    def _iterate(self, w, A, B, wc=None, C=None):
        if self._coidx is None:
            self._coidx = np.ones(self._l_obs.flatten().shape, dtype=bool)

        M = inv(B[:,self._coidx] @ self._Q_ll[self._coidx,:][:,self._coidx] @ B[:,self._coidx].T)

        def check_inversion(N):
            if cond(N) < sys.float_info.epsilon or abs(det(N)) < sys.float_info.epsilon:
                raise ValueError("matrix 'N' singular or bad conditioned (det: {:.3f} | cond: {:.3f})".format(cond(N),
                                                                                                              abs(det(
                                                                                                                  N))))

        if C is not None and wc is not None:
            N = np.block([[-A.T @ M @ A, C.T],
                          [C, np.zeros((self.model._c, self.model._c))]])
        else:
            N = A.T @ M @ A

        # check N condition or singularity
        check_inversion(N)

        if C is not None and wc is not None:
            xkc = np.linalg.solve(N, np.block([[A.T @ M @ w], [-wc]]))
            self._dx = xkc[:self.model._u]
        else:
            self._dx = - np.linalg.solve(N, A.T @ M @ w)

        k = - M @ (A @ self._dx + w)

        # residuals
        # v = self._Q_ll[self._coidx, self._coidx] @ B[:, self._coidx].T @ k
        v = self._Q_ll @ B.T @ k
        v[~self._coidx] = 0.

        # compute variance aposteriori
        self._f = (self._r*self.model._q - self.model._u + self.model._c)
        self._var0_post = (-k.T @ (w + A @ self._dx) / self._f)[0,0]
        self._vtPv = self._var0_post * self._f

        self._v = v.reshape((-1, self._l_obs.shape[1]))

        # update observations
        self._l = self._l_obs + self._v
        if self.model._auxiliary:
            self._l = self.model._auxiliary(self._l)

        if self._options['journal']:
            self._l_journal[self._iterations - 1, :self._n] = self._l.flatten()
            self._l_journal[self._iterations - 1, self._n:] = self._v.flatten()
            self._x_journal[self._iterations - 1, :self.model._u] = self._x.flatten() + self._dx.flatten()
            self._x_journal[self._iterations - 1, self.model._u:] = self._dx.flatten()

        if self._options['verbose']:
            self._print_iteration()

        # update parameters
        self._x += self._dx.flatten()

        return v, N, M

    def _hauptprobe(self):
        w = []
        for i, l in enumerate(self._l):
            if self.model._sym_c:
                w += [self.model._model(*self._x, *l, *self._const[i, :])]
            else:
                w += [self.model._model(*self._x, *l)]
        w = np.vstack(w)
        if np.max(np.abs(w)) < self._options['epsilon']:
            return True, w.flat[np.abs(w).argmax()]
        else:
            return False, w.flat[np.abs(w).argmax()]

    def _compute_varcovar(self, A, B, N, M):
        # accuracy of parameters
        self._Q_xx = inv(N)[:self.model._u, :self.model._u]
        # correlations of parameters
        self._R_xx = compute_corr_coeff(self._Q_xx)

        # accuracy of observations
        Qkk = M - M @ A @ self._Q_xx @ A.T @ M
        self._Q_vv = self._Q_ll @ B.T @ Qkk @ B @ self._Q_ll
        self._Q_llp = self._Q_ll - self._Q_vv

        self._Sigma_xx = self._Q_xx * self._var0_post
        self._Sigma_vv = self._Q_vv * self._var0_post
        self._Sigma_llp = self._Q_llp * self._var0_post

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
        if np.all(self._coidx):
            assert np.round(np.trace(self._Q_vv @ inv(self._Q_ll))) == self._f

        T = self._var0_post / self._var0_prio
        F = (chi2.ppf(self._options['alpha']/2, self._f)/self._f, chi2.ppf(1-self._options['alpha']/2, self._f)/self._f,
             chi2.ppf(1-self._options['alpha'], self._f)/self._f)

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


