"""
Multi-armed bandits
"""
import numpy as np
from scipy.special import expit


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
    " K-armed contextual bandit with binary payoffs "
    def initialize(self, **kwargs):
        d = kwargs.get("d")  # dimension of feature vectors
        loc = kwargs.get("loc", 0)
        scale = kwargs.get("scale", 1)
        # each arm has its own feature vector of length d
        self.features = np.random.normal(size=self.K * d).\
                reshape((self.K, d))
        # parameters are N(loc, scale) using specified parameters
        # default: standard normal
        theta = np.random.normal(loc=loc,
                                 scale=scale,
                                 size=self.K * d).\
               reshape((self.K, d))
        self.parameters = theta


    def pull_arm(self, k):
        x = self.get_features(k)  # d-vector
        theta = self.parameters[k, :]  # d-vector
        mu = expit(np.dot(x.T, theta))  # scalar
        return np.random.binomial(n=1,
                                  p=mu,
                                  size=1)[0]

    def get_features(self, k):
        " get features for the specified arm "
        return self.features[k, :]

    def get_mean(self):
        mu = np.zeros(self.get_K())
        for k, _ in enumerate(mu):
            x = self.get_features(k)  # d-vector
            theta = self.parameters[k, :]  # d-vector
            mu[k] = np.dot(x.T, theta)  # scalar
        return expit(mu)
