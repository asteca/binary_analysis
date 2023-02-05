
import numpy as np
from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def bruteForce(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    # gamma_q_lst = np.linspace(.2, 10, 10)
    aa = np.linspace(.0, 1, 5)
    gamma_q_lst = (aa[1:] + aa[:-1]) * .5
    q_step = (gamma_q_lst[1] - gamma_q_lst[0]) * .5

    # ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])

    lkl_old, gamma_q_opt = 0, []
    # with alive_bar(100 * 1000) as bar:
    for gamma_q1 in gamma_q_lst:
        for gamma_q2 in gamma_q_lst:
            for gamma_q3 in gamma_q_lst:
                for gamma_q4 in gamma_q_lst:
                    ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])
                    gamma_q = (q_step, gamma_q1, gamma_q2, gamma_q3, gamma_q4)
                    lkl = minFunc(
                        gamma_q, q_dist, clust_synth, obs_uncert, cluster_lkl,
                        masses_mask, ran_x, False)
                    if lkl > lkl_old:
                        gamma_q_opt = gamma_q
                        lkl_old = lkl
                    # bar()

    IMF_lkl_params = gamma_q_opt[1:]
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

    # Sample the 'q' values used for the masses
    # q_step = gamma_q[0]
    # q_vals = []
    # for i, msk in enumerate(masses_mask):
    #     q = ran_x[msk]
    #     msk2 = ((gamma_q[i + 1] - q_step) < q) & (q <= gamma_q[i + 1] + q_step)
    #     q[~msk2] = 0
    #     q_vals += list(q)
    # q_vals = np.array(q_vals)

    q_step = gamma_q[0]
    q_vals = []
    for i, msk in enumerate(masses_mask):
        q = ran_x * 1
        msk2 = ((gamma_q[i + 1] - q_step) < q) & (q <= gamma_q[i + 1] + q_step)
        q[~msk2] = 0
        q_vals += list(q[msk])
    q_vals = np.array(q_vals)

    # import matplotlib.pyplot as plt
    # plt.hist(q_vals);plt.show()

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
