
import numpy as np
# from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import gamma_q_min, gamma_q_max
from modules.methods import method0, method1, method2, method3, method4,\
    method5, method6, method7, method8


def bruteForce(MM, q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
    """
    """
    N_gamma = 10
    gamma_q_lst = np.linspace(gamma_q_min, gamma_q_max, N_gamma)

    if MM == "M0":
        qValsSample = method0
    elif MM == "M1":
        qValsSample = method1
    elif MM == "M2":
        qValsSample = method2
    elif MM == "M3":
        gamma_q_lst_N = (np.linspace(0.01, .99, 20))
        qValsSample = method3
    elif MM == "M4":
        qValsSample = method4
    elif MM == "M5":
        gamma_q_lst = np.linspace(0., 1, 10)
        qValsSample = method5
    elif MM == "M6":
        qValsSample = method6
    elif MM == "M7":
        gamma_q_lst_N = (np.linspace(0.01, .99, 20))
        qValsSample = method7
    elif MM == "M8":
        gamma_q_lst_N1 = (np.linspace(0.01, .9, 10))
        gamma_q_lst_N2 = (np.linspace(2, 5, 5))
        qValsSample = method8

    def coreFunc(lkl_old, gamma_q_opt, gamma_q):
        ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])
        lkl = minFunc(
            qValsSample, gamma_q, q_dist, clust_synth, obs_uncert,
            cluster_lkl, masses_mask, ran_x, False)
        if lkl > lkl_old:
            gamma_q_opt = gamma_q
            lkl_old = lkl

        return lkl_old, gamma_q_opt

    lkl_old, gamma_q_opt = 0, []

    if MM in "M0":
        lkl_old, gamma_q_opt = coreFunc(lkl_old, gamma_q_opt, np.nan)

    if MM in ("M1", "M4", "M5"):
        for gamma_q1 in gamma_q_lst:
            for gamma_q2 in gamma_q_lst:
                for gamma_q3 in gamma_q_lst:
                    for gamma_q4 in gamma_q_lst:
                        gamma_q = (gamma_q1, gamma_q2, gamma_q3, gamma_q4)
                        lkl_old, gamma_q_opt = coreFunc(
                            lkl_old, gamma_q_opt, gamma_q)

    if MM in ("M2", "M6"):
        for gamma_q1 in gamma_q_lst:
            for gamma_q2 in gamma_q_lst:
                gamma_q = (gamma_q1, gamma_q2)
                lkl_old, gamma_q_opt = coreFunc(lkl_old, gamma_q_opt, gamma_q)

    if MM in ("M3", "M7"):
        for gamma_q1 in gamma_q_lst_N:
            for gamma_q2 in gamma_q_lst:
                for gamma_q3 in gamma_q_lst:
                    gamma_q = (gamma_q1, gamma_q2, gamma_q3)
                    lkl_old, gamma_q_opt = coreFunc(
                        lkl_old, gamma_q_opt, gamma_q)

    if MM == "M8":
        for gamma_q1 in gamma_q_lst_N1:
            for gamma_q2 in gamma_q_lst_N2:
                for gamma_q3 in gamma_q_lst:
                    for gamma_q4 in gamma_q_lst:
                        for gamma_q5 in gamma_q_lst:
                            gamma_q = (gamma_q1, gamma_q2, gamma_q3, gamma_q4,
                                       gamma_q5)
                            lkl_old, gamma_q_opt = coreFunc(
                                lkl_old, gamma_q_opt, gamma_q)

    IMF_lkl_params = gamma_q_opt
    ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])
    IMF_synth_clusts = minFunc(
        qValsSample, gamma_q_opt, q_dist, clust_synth, obs_uncert, cluster_lkl,
        masses_mask, ran_x, True)

    return lkl_old, IMF_lkl_params, IMF_synth_clusts


def minFunc(
    qValsSample, gamma_q, q_dist, clust_synth, obs_uncert, cluster_lkl,
        masses_mask, ran_x, ret_clust, BF_DE='BF'):
    """
    """
    # Using the 'gamma_p' values obtain the 'q' values for each mass
    q_vals = qValsSample(gamma_q, masses_mask, ran_x)

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


# def diffEvol(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_mask):
#     """
#     """
#     ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])

#     bounds = [(gamma_q_min, gamma_q_max), (gamma_q_min, gamma_q_max),
#               (gamma_q_min, gamma_q_max), (gamma_q_min, gamma_q_max)]
#     res = DE(minFunc, bounds, tol=DE_tol, args=(
#         q_dist, clust_synth, obs_uncert, cluster_lkl, masses_mask, ran_x,
#         False, 'DE'))

#     IMF_lkl_params = res.x
#     IMF_synth_clusts = minFunc(
#         IMF_lkl_params, q_dist, clust_synth, obs_uncert, cluster_lkl,
#         masses_mask, ran_x, True, 'DE')

#     return res.fun, IMF_lkl_params, IMF_synth_clusts
