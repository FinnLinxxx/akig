import numpy as np
import sympy as sp


class AdjustmentConfig:
    def __init__(self):
        pass


class GHModelConfig:
    def __init__(self, params=None, obs=None):

        self.title = None
        self.description = None

        self.model_short_name = 'GH'
        self.model_name = 'Gauss-Helmert-Model'

        # Symbolic variables (unknowns x, observations l and model phi(x,l) = 0
        self._sym_x = None
        self._sym_l = None
        self._sym_c = None  # constants
        self._sym_phi = None
        self._sym_A = None
        self._sym_B = None
        # models for additional constraints (between parameters)
        self._sym_gamma = None
        self._sym_C = None

        # store the callables
        self._model = None
        self._model_A = None
        self._model_B = None
        self._constraint = None
        self._constraint_C = None
        self._auxiliary = None  # auxiliary method to call after l = l + v

        self._m = 0  # number of different observations
        self._q = 0  # number of equations per observation row
        self._c = 0  # number of constraints
        self._u = 0  # number of parameters

        # add params conifg
        self.params_config = params
        # add obs config
        self.obs_config = obs

    def _add_model(self, phi, x, l, c=None):
        self._sym_phi = phi
        self._sym_x = list(x)
        self._sym_l = list(l)
        if c is not None:
            self._sym_c = list(c)

        self._u = len(x)
        self._m = len(l)
        self._q = phi.shape[0]

    def add_model(self, phi, x, l, phix, Ax, Bx, c=None):

        self._add_model(phi, x, l, c)

        self._model = phix
        self._model_A = Ax
        self._model_B = Bx

    def add_model_autodiff(self, phi, x, l, c=None):
        """
        Define the model and initialize

        :param phi: should be sympy.Matrix
        :param x: parameters (list of symbols)
        :param l: observations (list of symbols)
        :param c: constants (list of symbols)
        :return: None
        """
        self._add_model(phi, x, l, c)

        if c:
            self._model = sp.lambdify(list(self._sym_x + self._sym_l + self._sym_c), self._sym_phi)
        else:
            self._model = sp.lambdify(list(self._sym_x + self._sym_l), self._sym_phi)

        # compute derivatives automatically using sympy
        self._compute_derivatives()

    def _add_constraint(self, gamma, gammax):
        self._sym_gamma = gamma
        self._constraint = gammax

    def add_constraint(self, gamma, gammax, Cx):
        self._constraint_C = Cx

        # dummy create C for shape
        x_dummy = np.zeros((self._u,))
        C_dummy = self._constraint_C(*x_dummy)

        self._add_constraint(gamma, gammax)
        self._c = self._sym_C.shape[0]

    def add_constraint_autodiff(self, gamma):
        gammax = sp.lambdify(list(self._sym_x), gamma)

        self._add_constraint(gamma, gammax)
        # compute derivatives automatically using sympy
        self._compute_derivatives()
        # number of constraints
        self._c = self._sym_C.shape[0]

        self._add_constraint(gamma, gammax)

    def get_model_lambdas(self):
        import dill
        dill.settings['recurse'] = True
        return dill.dumps(self._model), dill.dumps(self._model_A), dill.dumps(self._model_B)

    def load_model_lambdas(self):
        import dill
        self._model = dill.loads(self._model)
        self._model_A, self._model_B = dill.loads(self._model_A), dill.loads(self._model_B)

    def get_syms(self):
        if self._sym_c is None:
            return self._sym_x, self._sym_l
        else:
            return self._sym_x, self._sym_l, self._sym_c

    def get_model_syms(self):
        if self._sym_gamma is None:
            return self._sym_phi, self._sym_A, self._sym_B
        else:
            return self._sym_phi, self._sym_A, self._sym_B, self._sym_gamma, self._sym_C

    def _compute_derivatives(self):
        """
        Compute derivatives using sympy and store callables for fast computation
        :return: None
        """
        self._sym_A = self._sym_phi.jacobian(sp.Matrix(self._sym_x))
        self._sym_B = self._sym_phi.jacobian(sp.Matrix(self._sym_l))

        if self._sym_c:
            params = list(self._sym_x + self._sym_l + self._sym_c)
        else:
            params = list(self._sym_x + self._sym_l)
        self._model_A = sp.lambdify(params, self._sym_A)
        self._model_B = sp.lambdify(params, self._sym_B)

        if self._sym_gamma:
            self._sym_C = self._sym_gamma.jacobian(sp.Matrix(self._sym_x))
            self._constraint_C = sp.lambdify(list(self._sym_x), self._sym_C)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            # first convert lambdas to dills
            # TODO: check if constraints
            self._model, self._model_A, self._model_B = self.get_model_lambdas()
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            config = pickle.load(f)
            # load dills
            # TODO: check if constraints
            config.load_model_lambdas()
            return config

    def show(self):
        import textwrap

        if self.title is not None:
            print('{0:30.20} ({1})'.format(self.title, self.model_short_name))
        else:
            print('{0:30.20} ({1})'.format('Unnamed', self.model_short_name))
        print('-'*35)
        print('')
        if self.description:
            print('\n'.join(textwrap.wrap(self.description, width=60, replace_whitespace=False)))
        else:
            print(' -- no description available -- ')


class GMModelConfig:
    def __init__(self, params=None, obs=None):

        self.title = None
        self.description = None

        self.model_short_name = 'GM'
        self.model_name = 'Gauss-Markov-Model'

        # Symbolic variables (unknowns x, observations l and model phi(x,l) = 0
        self._sym_x = []
        self._sym_l = []
        self._sym_c = []  # constants
        self._sym_phi = None
        self._sym_A = None

        # store adjustment blocks
        self._blocks = []

        # store the callables
        self._model = None
        self._model_A = None
        self._auxiliary = None  # auxiliary method to call after l = l + v

        self._m = 0  # number of different observations
        self._n = 0  # number of total observations
        self._r = 0  # number of rows (observations)
        self._t = 0  # number of constants
        self._u = 0  # number of parameters

        # add params conifg
        self.params_config = params
        # add obs config
        self.obs_config = obs

    def _add_model(self, block, x, c=None):
        self._blocks.append(block)

        # update parameters and constants
        self._sym_x = list(dict.fromkeys(self._sym_x + list(x)))
        if c is not None:
            self._sym_c = list(dict.fromkeys(self._sym_c + list(c)))
            # update number of constants
            self._t = len(self._sym_c)

        # update symbolic observations
        ls = []
        for b in self._blocks:
            ls += b._sym_l
        self._sym_l = list(set(ls))

        # update number of parameters
        self._u = len(self._sym_x)

        self._compute_derivatives()

        # # initialize journal data
        # if self._options['journal']:
        #     self._x_journal = np.zeros((self._options['max_iterations'], self._u * 2))

    def add_model(self, phi, x, l, phix, Ax, c=None, name='default'):
        """
        Define the model and initialize (same as add_model_autodiff())

        :param name: name of the model block (unique identifier)
        :param phi: should be sympy.Matrix
        :param x: parameters (list of symbols)
        :param l: observations (list of symbols)
        :param phix: callable of the model
        :param Ax: callable of jacobian
        :param c: constants (list of symbols)
        :return: None
        """
        block = AdjustBlock(self, name, phi, l, auto_diff=False)
        block._model = phix
        block._model_A = Ax

        self._add_model(block, x, c)

    def add_model_autodiff(self, phi, x, l, c=None, name='default'):
        """
        Define the model and initialize

        :param name: name of the model block (unique identifier)
        :param phi: should be sympy.Matrix
        :param x: parameters (list of symbols)
        :param l: observations (list of symbols)
        :param c: constants (list of symbols)
        :return: None
        """
        block = AdjustBlock(self, name, phi, l)
        self._add_model(block, x, c)

    def _compute_derivatives(self):
        """
        Compute derivatives using sympy and store callables for fast computation
        :return: None
        """
        # compute derivatives automatically using sympy
        for b in self._blocks:
            b.compute_lambdas()

    def get_syms(self):
        if self._sym_c is None:
            return self._sym_x, self._sym_l
        else:
            return self._sym_x, self._sym_l, self._sym_c

    def get_model_syms(self):
        return self._sym_phi, self._sym_A

    def set_solver(self, solver):
        for b in self._blocks:
            b.set_solver(solver)


class AdjustBlock:
    def __init__(self, model, name, phi, l, auto_diff=True):
        # identifier of the block
        self.name = name

        # reference of the model
        self.model = model
        # and reference of the solver
        self.solver = None

        # model, observations and jacobian
        self._sym_phi = phi
        # number of equations per observation row
        self._q = phi.shape[0]
        self._sym_l = list(l)
        self._sym_A = None

        self._obs_row_names = None

        self._m = len(self._sym_l)  # number of observations per row
        self._r = 0  # number of rows (observations)

        self._n = 0  # total number of observations
        self._model = None  # lambdified model
        self._model_A = None  # lambdified design matrix
        self._autodiff = auto_diff   # automatic differentiation using sympy

        # observed measurements
        self._l_obs = None
        self._Sigma_ll = None
        self._Q_ll = None

        self._l = None
        self._v = None

        # self.compute_derivatives(params)

    def set_solver(self, solver):
        self.solver = solver

    def compute_lambdas(self):
        """
        Compute derivatives using sympy and store callables for fast computation
        :return: None
        """
        if self.model._sym_c:
            params = list(self.model._sym_x + self._sym_l + self.model._sym_c)
        else:
            params = list(self.model._sym_x + self._sym_l)

        if self._autodiff:
            self._model = sp.lambdify(params, self._sym_phi)
            self._sym_A = self._sym_phi.jacobian(sp.Matrix(self.model._sym_x))
            self._model_A = sp.lambdify(params, self._sym_A)

    def set_stochastic_model(self, sigma_ll=None, var0=1.):
        self.solver._var0_prio = var0
        if sigma_ll is None:
            # self._Sigma_ll = dia_matrix((np.array([1] * self._n), [0]), shape=(self._n, self._n))
            self._Sigma_ll = np.eye(self._n)
        else:
            if sigma_ll.shape != (self._n, self._n):
                raise ValueError("invalid dimension for parameter 'sigma_ll'. (should be {0:d}x{0:d})".format(self._n))
            self._Sigma_ll = sigma_ll

        self._Q_ll = 1 / self.solver._var0_prio * self._Sigma_ll

    def add_data(self, data, sigma_ll=None, var0=1., row_names=None):
        data = np.atleast_2d(data)
        if data.shape[1] != self._m:
            raise ValueError('wrong number of observations in supplied data. (should be {0:d})'.format(self._m))
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

        # initialize stochastic model
        self.set_stochastic_model(sigma_ll, var0)

    def recompute_matrices(self):
        dl, A = [], []

        for i, l in enumerate(self._l_obs):
            if self.model._sym_c:
                dl += [l[:, None] - self._model(*self.solver._x, *l, *self.solver._const[i, :])]
                A += [self._model_A(*self.solver._x, *l, *self.solver._const[i, :])]
            else:
                dl += [l[:, None] - self._model(*self.solver._x, *l)]
                A += [self._model_A(*self.solver._x, *l)]

        A = np.vstack(A)
        dl = np.vstack(dl)
        return dl, A

    def get_model_lambdas(self):
        import dill
        dill.settings['recurse'] = True
        return dill.dumps(self._model), dill.dumps(self._model_A)

    def load_model_lambdas(self):
        import dill
        self._model = dill.loads(self._model)
        self._model_A = dill.loads(self._model_A)

    def _hauptprobe(self):
        w = []
        for i, l in enumerate(self._l):
            if self.model._sym_c:
                w += [l[:, None] - self._model(*self.solver._x, *l, *self.solver._const[i, :])]
            else:
                w += [l[:, None] - self._model(*self.solver._x, *l)]
        w = np.vstack(w)
        return w

