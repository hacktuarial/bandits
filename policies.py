"""
Variety of policies for multi-armed bandits
"""
import numpy as np
from utils import softmax


def _quadratic_form(A, x):
    " x'Ax "
    return np.dot(np.dot(x.T, A), x)


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


def lin_ucb_disjoint_policy(bandit,
                            bandit_context,
                            n_plays,
                            alpha,
                            play_all_first=False,
                            l2_penalty=1.0):
    """
    source:
    Algorithm 1 of https://arxiv.org/pdf/1003.0146.pdf
    """
    K = bandit.get_K()
    # keep track of decisions made and rewards
    path = np.empty(n_plays, dtype=int)
    rewards = [np.nan] * n_plays
    # parameters for the policy
    d = len(bandit.get_features())
    A = _initialize_A(K=K, d=d)
    theta = np.zeros((K, d))
    b = np.zeros((K, d))
    p = np.zeros(K)

    for t in range(n_plays):
        x = bandit.get_features()  # iid context
        if t < K and play_all_first:
            k = t
        else:
            for k in range(K):
                theta[k, :] = np.linalg.solve(A[k, :, :], b[k, :])
                exploit = np.dot(theta[k, :].T, x)
                A_inv = np.linalg.inv(A[k, :, :])
                explore = np.sqrt(_quadratic_form(A_inv, x))
                p[k] = exploit + alpha * explore
            k = np.argmax(p)
        reward = bandit.pull_arm(k)
        # update our parameters
        A[k, :, :] += np.dot(x, x.T)
        b[k, :] += reward * x
        # book keeping
        path[t] = k
        rewards[t] = reward

    return (path, rewards, theta)


def _initialize_A(K, d):
    """
    """
    A = np.zeros(d * d * K).reshape((K, d, d))
    for k in xrange(K):
        for i in xrange(d):
            A[k, i, i] = 1
    return A


def lin_ucb_hybrid_policy(bandit,
                          bandit_context,
                          n_plays,
                          alpha,
                          play_all_first=False):
    """
    source:
    Algorithm 2 of https://arxiv.org/pdf/1003.0146.pdf
    """
    dim_shared = bandit_context.dim_shared
    dim_each = bandit_context.dim_each
    # A_0, b_0 are for the features shared by all arms
    A_0 = np.eye(bandit.get_K())
    b_0 = np.zeros(bandit.get_K())
    # arm-specific design matrices and rewards
    A = _initialize_A(K=bandit.get_K(),
                      d=dim_each)
    B = np.zeros((bandit.get_K(), dim_each, dim_shared))
    b = np.zeros((bandit.get_K(), dim_each))
    s = np.zeros((n_plays, bandit.get_K()))
    p = np.zeros((n_plays, bandit.get_K()))
    theta = np.zeros((bandit.get_K(), dim_each))
    # keep track of decisions made and rewards
    path = np.empty(n_plays, dtype=int)
    rewards = [np.nan] * n_plays
    for t in xrange(n_plays):
        # generate features
        if t < bandit.get_K() and play_all_first:
            k = t
        else:
            # observe features of all arms
            x = bandit.get_individual_features(k)
            z = bandit.get_shared_features(k)
            beta_hat = np.linalg.solve(A_0, b_0)
            for k in xrange(bandit.get_K()):
                theta[k, :] = np.linalg.solve(A[k, :, :],
                                              b[k, :] - np.dot(B[k, :, :],
                                                               beta_hat))
                A_0_inv = np.linalg.inv(A_0)
                A_k_inv = np.linag.inv(A[k, :, :])
                s[t, k] = _quadratic_form(A_0_inv, z)
                s[t, k] -= 2 * np.dot(np.dot(z.T, A_0_inv),
                                      np.dot(B[k, :, :].T, A_k_inv, x))
                s[t, k] += _quadratic_form(A_k_inv, x)
                something_ugly = np.dot(np.dot(A_k_inv, B[k, :, :]),
                                        np.dot(A_0_inv, B[k, :, :].T),
                                        A_k_inv)
                s[t, k] += _quadratic_form(something_ugly, x)
                p[t, k] = np.dot(z.T, beta_hat) + np.dot(x.T, theta[k, :])
                p[t, k] += alpha * np.sqrt(s[t, k])
            k = np.argmax(p)
        reward = bandit.pull_arm(k)
        A_k_inv = np.linalg.inv(A[k, :, :])
        A_0 += _quadratic_form(A_k_inv, B[k, :, :])
        b_0 += np.dot(np.dot(B[k, :, :].T, A_k_inv), b[k, :])
        A[k, :, :] += np.dot(x, x.T)
        # recompute inverse
        A_k_inv = np.linalg.inv(A[k, :, :])
        B[k, :, :] += np.dot(x, z.T)
        b[k, :] += reward * x
        A_0 += np.dot(z, z.T) - _quadratic_form(B[k, :, :], A_k_inv)
        b_0 += reward * z - np.dot(np.dot(B[k, :, :].T, A_k_inv), b[k, :])
    return (path, rewards, beta_hat, theta)
