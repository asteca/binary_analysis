
import numpy as np
from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def bruteForce(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    N_gamma = 10
    gamma_q_lst = np.linspace(gamma_q_min, gamma_q_max, N_gamma)

    # ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])

    lkl_old, gamma_q_opt = 0, []
    # with alive_bar(100 * 1000) as bar:
    for gamma_q1 in gamma_q_lst:
        for gamma_q2 in gamma_q_lst:
            for gamma_q3 in gamma_q_lst:
                for gamma_q4 in gamma_q_lst:
                    ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])
                    gamma_q = (gamma_q1, gamma_q2, gamma_q3, gamma_q4)
                    lkl = minFunc(
                        gamma_q, q_dist, clust_synth, obs_uncert, cluster_lkl,
                        masses_mask, ran_x, False)
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
    ran_x.shape   : N_stars_q
    gamma_q.shape : Nq
    q_vals.shape  : N_stars (sum(N_stars_q))
    """

    # This is the method that *actually* assigns 'q' values sampling the
    # various power laws respecting the 'masses_mask' order.
    #
    # It gives inferior results compared to the approach above
    #
    q_vals = 1 * ran_x
    for i, msk in enumerate(masses_mask):
        q_vals[msk] = q_vals[msk]**(1 / gamma_q[i])

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
