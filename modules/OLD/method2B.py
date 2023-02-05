
import numpy as np
from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def bruteForce(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    N_gamma = 10
    gamma_q_lst = np.linspace(gamma_q_min, gamma_q_max, N_gamma)

    gamma_q_lst_N = (np.linspace(0.01, .99, 20))  # 1
    gamma_q_lst = np.linspace(0.01, 10, 20)

    lkl_old, gamma_q_opt = 0, []
    # from alive_progress import alive_bar
    # with alive_bar(N_gamma ** 4) as bar:
    for gamma_q1 in gamma_q_lst:
        for gamma_q2 in gamma_q_lst:
            # for gamma_q3 in gamma_q_lst:
            #     for gamma_q4 in gamma_q_lst:
            ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])
            gamma_q = (gamma_q1, gamma_q2, gamma_q3) #, gamma_q4)
            lkl = minFunc(
                gamma_q, q_dist, clust_synth, obs_uncert,
                cluster_lkl, masses_mask, ran_x, False)
            if lkl > lkl_old:
                gamma_q_opt = gamma_q
                lkl_old = lkl
            # bar()

    IMF_lkl_params = gamma_q_opt
    IMF_synth_clusts = minFunc(
        gamma_q_opt, q_dist, clust_synth, obs_uncert, cluster_lkl, masses_mask,
        ran_x, True)

    return lkl_old, IMF_lkl_params, IMF_synth_clusts


def diffEvol(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])

    bounds = [(gamma_q_min, gamma_q_max), (gamma_q_min, gamma_q_max),
              (gamma_q_min, gamma_q_max), (gamma_q_min, gamma_q_max)]
    res = DE(minFunc, bounds, tol=DE_tol, args=(
        q_dist, clust_synth, obs_uncert, cluster_lkl, masses_mask, ran_x,
        False, 'DE'))

    IMF_lkl_params = res.x
    IMF_synth_clusts = minFunc(
        IMF_lkl_params, q_dist, clust_synth, obs_uncert, cluster_lkl,
        masses_mask, ran_x, True, 'DE')

    return res.fun, IMF_lkl_params, IMF_synth_clusts


def InvPowerLaw(gamma_q, masses_mask, ran_x):
    """
    Sample the 'q' values used for the masses

    ran_x.shape   : N_stars_q
    gamma_q.shape : Nq
    q_vals.shape  : N_stars (sum(N_stars_q))
    """
    msum = masses_mask[0].sum()+masses_mask[1].sum()+masses_mask[2].sum()+masses_mask[3].sum()
    if msum != len(ran_x):
        print("M2B: inconsistent length")
        breakpoint()

    # # This approach fits the percentage that determines how many stars are
    # # sampled with gamma_q[1] and how many with gamma_q[2]. It does not take
    # # their masses into account meaning that the 'q' assignment is random
    # #
    # N = int(gamma_q[0] * len(ran_x))
    # q1 = ran_x[:N]**(1 / gamma_q[1])
    # q2 = ran_x[N:]**(1 / gamma_q[2])
    # q_vals = np.array(list(q1) + list(q2))

    # This approach uses the 'masses_mask' to split into two arrays: one with
    # the lowest masses and the other with the largest masses.
    msk0 = masses_mask[0]
    m123 = masses_mask[1] | masses_mask[2] | masses_mask[3]
    q1 = ran_x[msk0]**(1 / gamma_q[1])
    q2 = ran_x[m123]**(1 / gamma_q[2])
    q_vals = np.array(list(q1) + list(q2))

    if q_vals.min() < 0 or q_vals.max() > 1:
        print("M2B: q_vals out of range")
        breakpoint()

    return q_vals


def minFunc(
    gamma_q, q_dist, clust_synth, obs_uncert, cluster_lkl, masses_mask, ran_x,
        ret_clust, BF_DE='BF'):
    """
    """
    # Using the 'gamma_p' values obtain the 'q' values for each mass
    q_vals = InvPowerLaw(gamma_q, masses_mask, ran_x)

    clust_synth_binar = binarFunc.addBinaries(
        q_dist, clust_synth, q_vals)

    clust_synth_final = uncertanties.addErrors(
        clust_synth_binar, obs_uncert)

    if ret_clust:
        return q_vals, clust_synth_final

    Lkl = likelihood.getLkl(cluster_lkl, clust_synth_final)

    if BF_DE == 'BF':
        return Lkl
    else:
        return -Lkl
