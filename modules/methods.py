
import numpy as np
# from scipy.optimize import differential_evolution as DE
# from . import binarFunc, uncertanties, likelihood
# from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def method0(gamma_q, masses_mask, ran_x):
    """
    Simply assigns random numbers in the [0, 1]
    """
    q_vals = 1. * ran_x

    return q_vals


def method1(gamma_q, masses_mask, ran_x):
    """
    Selects groups of random numbers in the [0, 1] range and obtains their
    (1/gamma_i) power. Each of these values is then added to the 'q_vals'
    list with no regard for the 'masses_mask' order.

    This method depends on the number of bins used to generate the mass
    intervals (used to obtain 'masses_mask')
    """
    q_vals = []
    for i, msk in enumerate(masses_mask):
        q_vals += list(ran_x[msk]**(1 / gamma_q[i]))
    q_vals = np.array(q_vals)

    return q_vals


def method2(gamma_q, masses_mask, ran_x):
    """
    This approach uses the 'masses_mask' to split into two arrays: one with
    the lowest masses and the other with the largest masses.
    """
    msk0 = masses_mask[0]
    m123 = masses_mask[1] | masses_mask[2] | masses_mask[3]
    q1 = ran_x[msk0]**(1 / gamma_q[0])
    q2 = ran_x[m123]**(1 / gamma_q[1])
    q_vals = np.array(list(q1) + list(q2))

    return q_vals


def method3(gamma_q, masses_mask, ran_x):
    """
    This approach fits the percentage that determines how many stars are
    sampled with gamma_q[1] and how many with gamma_q[2]. It does not take
    their masses into account meaning that the 'q' assignment is random
    """
    N = int(gamma_q[0] * len(ran_x))
    q1 = ran_x[:N]**(1 / gamma_q[1])
    q2 = ran_x[N:]**(1 / gamma_q[2])
    q_vals = np.array(list(q1) + list(q2))

    return q_vals


def method4(gamma_q, masses_mask, ran_x):
    """
    This is the method that *actually* assigns 'q' values sampling the
    various power laws respecting the 'masses_mask' order.
    """
    for i, msk in enumerate(masses_mask):
        ran_x[msk] = ran_x[msk]**(1 / gamma_q[i])

    return ran_x


def method5(gamma_q, masses_mask, ran_x, STDDEV=.25):
    """
    """
    N_all = len(ran_x)
    q_vals = []
    for i, msk in enumerate(masses_mask):
        q = np.random.normal(gamma_q[i], STDDEV, N_all)
        q_vals += list(q[msk])
    q_vals = np.clip(q_vals, a_min=0., a_max=1)

    return q_vals


def method6(gamma_q, masses_mask, ran_x, STDDEV=.25):
    """
    """
    N_all = len(ran_x)
    msk0 = masses_mask[0]
    m123 = masses_mask[1] | masses_mask[2] | masses_mask[3]
    q1 = np.random.normal(gamma_q[0], STDDEV, N_all)
    q2 = np.random.normal(gamma_q[1], STDDEV, N_all)
    q_vals = np.array(list(q1[msk0]) + list(q2[m123]))
    q_vals = np.clip(q_vals, a_min=0., a_max=1)

    return q_vals


def method7(gamma_q, masses_mask, ran_x):
    """
    https://stackoverflow.com/a/71363630/1391441
    """
    msk = ran_x < gamma_q[0]
    ran_x[msk] = ran_x[msk]**(1 / gamma_q[1])
    ran_x[~msk] = ran_x[~msk]**(1 / gamma_q[2])

    return ran_x


def method8(gamma_q, masses_mask, ran_x):
    """
    https://stackoverflow.com/a/71363630/1391441
    """
    import matplotlib.pyplot as plt
    plt.show()
    plt.hist(ran_x);plt.show()
    x1 = gamma_q[0]
    msk1 = ran_x < x1
    x2 = gamma_q[0] + (1 - x1) / gamma_q[1]
    msk2 = (ran_x >= x1) & (ran_x < x2)
    # msk3 = ran_x >= x2

    ran_x[msk1] = ran_x[msk1]**(1 / gamma_q[2])
    ran_x[msk2] = ran_x[msk2]**(1 / gamma_q[3])
    # ran_x[msk3] = ran_x[msk3]**(1 / gamma_q[4])
    plt.hist(ran_x);plt.show()
    breakpoint()

    return ran_x
