"""
Utility functions for multi-armed bandits
"""
import numpy as np


def regret(bandit, path):
    """ calculate regret = reward under optimal policy
    minus expected value of the actual policy taken
    only for bandits whose parameters == expected value of reward"""
    optimal = max(bandit.get_mean())
    expected = bandit.get_mean()[path]
    return np.mean(optimal - expected)


def softmax(x):
    " return softmax of an array "
    return np.exp(x) / sum(np.exp(x))


def cummean(x):
    " cumulative mean of an array "
    return [np.mean(x[:i+1]) for i in range(len(x))]


def moving_average(x, d):
    " moving average of x with window-size d"
    # TODO: test this
    vec = np.zeros(len(x) - d)
    for i in range(len(vec)):
        vec[i] = np.mean(x[i:i+d])
    return vec

