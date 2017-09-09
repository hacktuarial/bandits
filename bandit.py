"""
Multi-armed bandits
"""
import abc
import numpy as np
# from scipy.special import expit


class Bandit(object):
    " Generic bandit "
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        " initialize "
        self.parameters = None
        self.initialize(**kwargs)

    @abc.abstractmethod
    def initialize(self, **kwargs):
        " create parameters for each arm "
        raise NotImplementedError

    @abc.abstractmethod
    def get_K(self):
        " return number of arms of this bandit "
        raise NotImplementedError

    def get_parameters(self):
        " return parameters of this bandit "
        return self.parameters

    @abc.abstractmethod
    def pull_arm(self, k, **kwargs):
        " play specified arm of this bandit "
        raise NotImplementedError

    @abc.abstractmethod
    def get_mean(self):
        " translate bandit features into expected rewards "
        raise NotImplementedError


class LinearBandit(Bandit):
    " K-armed bandit with Gaussian payoffs "
    def initialize(self, **kwargs):
        self.noise = kwargs.get('noise')
        self.parameters = kwargs.get('parameters')

    def pull_arm(self, k):
        return np.random.normal(loc=self.parameters[k],
                                scale=self.noise,
                                size=1)[0]

    def get_K(self):
        return len(self.get_parameters())

    def get_mean(self):
        return self.get_parameters()


class BinaryBandit(Bandit):
    """
    context-free bernoulli bandit with K arms.
    each arm pays 1 with some probability p
    or 0 with probability 1-p
    """
    def initialize(self, **kwargs):
        self.parameters = kwargs.get('parameters')

    def pull_arm(self, k):
        return np.random.binomial(n=1,
                                  p=self.parameters[k],
                                  size=1)[0]

    def get_mean(self):
        return self.get_parameters()

    def get_K(self):
        return len(self.get_parameters())


class LinearDisjointContextualBandit(Bandit):
    """
    Reward is a dot product of a context-vector and
    the selected arm's parameters.
    Rows do not share any parameters.
    These parameters are random normal with user supplied
    location and scale. Default: standard normal
    """
    def initialize(self, **kwargs):
        " parameters must be a K x d array "
        self.parameters = kwargs.get('parameters')
        self.noise = kwargs.get('noise')

    def pull_arm(self, **kwargs):
        """
        :param: k. which arm to pull
        :param: x. vector summarising context
        """
        k = kwargs.get('k')
        context_vector = kwargs.get('x')
        exp_rwd = np.dot(self.parameters[k, :].T, context_vector)
        noise = np.random.normal(loc=0, scale=self.noise, size=1)[0]
        reward = exp_rwd + noise
        if reward > 0:
            return 1
        else:
            return 0

    def get_K(self):
        return self.get_parameters().shape[0]

    def get_mean(self):
        " mean depends on the context vector "
        raise NotImplementedError


class LinearHybridContextualBandit(Bandit):
    """
    Reward is a dot product of a context-vector and
    the selected arm's parameters.
    Rows share some parameters.
    These parameters are random normal with user supplied
    location and scale. Default: standard normal
    """
    def initialize(self, **kwargs):
        """
        :param: common_parameters: vector of parameters shared by all arms
        :param: arm_parameters. array of parameters, one row for each arm
        """
        self.common_parameters = kwargs.get('common_parameters')
        self.arm_parameters = kwargs.get('arm_parameters')
        self.noise = kwargs.get('noise')

    def pull_arm(self, **kwargs):
        """
        :param: k. which arm to pull
        :param: x. vector of user/arm features,
            with parameters specific to arm k
        :param: z. vector of user/arm features,
            with parameters shared by all arms
        """
        which_arm = kwargs.get('k')
        context_x = kwargs.get('x')
        context_z = kwargs.get('z')
        exp_rwd = np.dot(self.arm_parameters[which_arm, :].T, context_x)
        exp_rwd += np.dot(self.common_parameters.T, context_z)
        reward = exp_rwd + np.random.normal(loc=0, scale=self.noise, size=1)[0]
        if reward > 0:
            return 1
        else:
            return 0

    def get_K(self):
        return self.arm_parameters.shape[0]

    def get_mean(self):
        " mean depends on the context vector "
        raise NotImplementedError
