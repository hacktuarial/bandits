"""
Variety of policies for multi-armed bandits
"""
import numpy as np
from utils import softmax


def random_policy(bandit, n_plays):
    "play every arm once, then play completely at random"
    assert n_plays >= bandit.K
    outcomes = np.zeros((bandit.K, 2))
    path = [np.nan] * n_plays
    rewards = [np.nan] * n_plays
    for t in range(n_plays):
        if t < bandit.K:
            k = t
        else:
            k = np.random.choice(range(0, bandit.K), size=1)[0]
        reward = bandit.pull_arm(k)
        outcomes[k, reward] += 1
        path[t] = k
        rewards[t] = reward
    # TODO: write tests to check this
    assert int(outcomes.sum()) == n_plays
    assert sum([np.isnan(x) for x in path]) == 0
    assert sum([np.isnan(x) for x in rewards]) == 0
    return (outcomes, path, rewards)


def epsilon_greedy_policy(bandit, n_plays, epsilon):
    " epsilon-greedy policy for multi-armed bandits "
    outcomes = np.zeros((bandit.K, 2))
    path = [np.nan] * n_plays
    rewards = [np.nan] * n_plays
    for t in range(n_plays):
        if t < bandit.K:
            k = t
        else:
            if np.random.uniform(size=1)[0] < epsilon:
                k = np.random.choice(range(0, bandit.K), size=1)[0]
            else:
                k = np.argmax(outcomes[:, 1] /
                              np.apply_along_axis(sum, 1, outcomes))
        reward = bandit.pull_arm(k)
        outcomes[k, reward] += 1
        path[t] = k
        rewards[t] = reward
    return (outcomes, path, rewards)


def boltzmann_policy(bandit, n_plays, temperature):
    """ temperature must be a function of a single parameter
    temperature(t: int) -> float """
    outcomes = np.zeros((bandit.K, 2))
    path = [np.nan] * n_plays
    rewards = [np.nan] * n_plays
    # play every arm once
    for t in range(n_plays):
        if t < bandit.K:
            k = t
        else:
            temp = temperature(t)
            weights = outcomes[:, 1] / np.apply_along_axis(sum, 1, outcomes)
            weights *= 1/temp
            k = np.random.choice(range(bandit.K),
                                 p=softmax(weights),
                                 size=1)[0]
        reward = bandit.pull_arm(k)
        outcomes[k, reward] += 1
        rewards[t] = reward
        path[t] = k
    return (outcomes, path, rewards)


def ucb1_policy(bandit, n_plays):
    " upper confidence bound policy for multi-armed bandit "
    outcomes = np.zeros((bandit.K, 2))
    path = [np.nan] * n_plays
    rewards = [np.nan] * n_plays
    for t in range(n_plays):
        if t < bandit.K:
            k = t
        else:
            successes = outcomes[:, 1]
            attempts = np.apply_along_axis(sum, 1, outcomes)
            exploit = successes / attempts
            explore = 2 * np.log(sum(attempts)) / attempts
            k = np.argmax(exploit + np.sqrt(explore))
        reward = bandit.pull_arm(k)
        outcomes[k, reward] += 1
        path[t] = k
        rewards[t] = reward
    return (outcomes, path, rewards)


def thompson_policy(bandit, n_plays):
    " Thompson sampling for a multi-armed bandit "
    outcomes = np.zeros((bandit.K, 2))
    path = [np.nan] * n_plays
    rewards = [np.nan] * n_plays
    for t in range(n_plays):
        samples = [np.random.beta(a=outcomes[k, 1] + 1,
                                  b=outcomes[k, 0] + 1,
                                  size=1)
                   for k in range(bandit.K)]
        k = np.argmax(samples)
        reward = bandit.pull_arm(k)
        outcomes[k, reward] += 1
        path[t] = k
        rewards[t] = reward
    return (outcomes, path, rewards)