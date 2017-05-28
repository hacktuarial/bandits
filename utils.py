"""
Utility functions for multi-armed bandits
"""
import numpy as np


def regret(true_params, outcomes):
    """ calculate regret = reward under optimal policy
    minus the actual policy taken """
    optimal = max(true_params) * outcomes.sum()
    successes = outcomes[:, 1].sum()
    return (optimal - successes) / outcomes.sum()


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

