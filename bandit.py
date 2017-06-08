"""
Multi-armed bandits
"""
import numpy as np
from scipy.special import expit
from scipy import stats


class Bandit(object):
    " Generic bandit "
    def __init__(self, K, **kwargs):
        " initialize "
        self.parameters = None
        self.K = K
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        " create parameters for each arm "
        pass

    def get_K(self):
        " return number of arms of this bandit "
        return self.K

    def get_parameters(self):
        " return parameters of this bandit "
        return self.parameters

    def pull_arm(self, k):
        " play specified arm of this bandit "
        pass

    def get_mean(self):
        " translate bandit features into expected rewards "
        pass


class LinearBandit(Bandit):
    " K-armed bandit with Gaussian payoffs "
    def initialize(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        noise = kwargs.get('sigma', 1)
        self.noise = noise
        self.parameters = np.random.normal(loc=loc,
                                           scale=scale,
                                           size=self.get_K())

    def pull_arm(self, k):
        return np.random.normal(loc=self.parameters[k],
                                scale=self.noise,
                                size=1)[0]

    def get_mean(self):
        raise NotImplementedError


class BinaryBandit(Bandit):
    """ bernoulli bandit with K arms.
    each arm pays 1 with some probability p
    or 0 with probability 1-p """
    def initialize(self, **kwargs):
        a = kwargs.get('a', 2)
        b = kwargs.get('b', 2)
        self.parameters = np.random.beta(a, b, size=self.K)

    def pull_arm(self, k):
        return np.random.binomial(n=1,
                                  p=self.parameters[k],
                                  size=1)[0]
    def get_mean(self):
        return self.get_parameters()


class DisjointLinearBandit(Bandit):
    " K-armed contextual bandit with 0/1 payoffs "
    def initialize(self, **kwargs):
        d = kwargs.get("d")  # dimension of feature vectors
        loc = kwargs.get("loc", 0)
        scale = kwargs.get("scale", 1)
        # parameters are N(loc, scale) using specified parameters
        # default: standard normal
        theta = np.random.normal(loc=loc,
                                 scale=scale,
                                 size=self.K * d).\
               reshape((self.K, d))
        self.parameters = theta
        # features are assumed to be standard normal
        # with a random covariance matrix
        # first one is an intercept
        self.features_mu = np.zeros(d - 1)
        self.features_Sigma = np.eye(d - 1)
        # more complex:
        # self.features_Sigma = stats.Wishart.rvs(df = d + 1,
        #                                         scale=1)


    def pull_arm(self, k):
        x = self.get_features()  # d-vector
        theta = self.parameters[k, :]  # d-vector
        mu = expit(np.dot(x.T, theta))  # scalar
        return np.random.binomial(n=1, p=mu, size=1)[0]

    def get_features(self):
        x = np.random.multivariate_normal(self.features_mu,
                                          self.features_Sigma,
                                          size=1)
        x = np.concatenate([np.ones(1), np.ravel(x)])
        return x

    def get_mean(self):
        return expit(self.parameters[:, 0])
