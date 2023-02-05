
import numpy as np
from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def bruteForce(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    gamma_q_lst = np.linspace(0., 1, 10)
    # aa = np.linspace(.0, 1, 5)
    # gamma_q_lst = (aa[1:] + aa[:-1]) * .5
    # q_step = (gamma_q_lst[1] - gamma_q_lst[0]) * .5

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

    # # Sample the 'q' values used for the masses
    # q_vals = []
    # for i, msk in enumerate(masses_mask):
    #     q = np.random.normal(gamma_q[i], .2, len(msk))
    #     q = np.clip(q, a_min=0., a_max=1)
    #     q_vals += list(q)
    # q_vals = np.array(q_vals)

    STDDEV = 0.1
    q1 = np.random.normal(gamma_q[0], STDDEV, 1000)
    q2 = np.random.normal(gamma_q[1], STDDEV, 150)
    q3 = np.random.normal(gamma_q[2], STDDEV, 50)
    N123 = len(q1) + len(q2) + len(q3)
    q4 = np.random.normal(gamma_q[3], STDDEV, len(ran_x) - N123)
    q_vals = np.array(list(q1) + list(q2) + list(q3) + list(q4))
    q_vals = np.clip(q_vals, a_min=0., a_max=1)

    if len(q_vals) != len(ran_x):
        print("q_vals shape mismatch")
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
