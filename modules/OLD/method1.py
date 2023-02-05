
import numpy as np
# from scipy.optimize import differential_evolution as DE
from . import binarFunc, uncertanties, likelihood
from modules.HARDCODED import mmin, mmax, gamma_m_min, gamma_m_max,\
    gamma_q_min, gamma_q_max


def bruteForce(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_interp):
    """
    """
    gamma_m_lst = np.linspace(gamma_m_min, gamma_m_max, 100)

    # Convert masses to [0, 1] range
    m_p = (masses_interp - mmin) / (mmax - mmin)

    ran_x = np.random.uniform(0, 1, clust_synth.shape[-1])

    lkl_old, gamma_m_opt = 0, 1000
    # with alive_bar(100 * 1000) as bar:
    for gamma_m in gamma_m_lst:
        # for _ in range(100):
        lkl = minFunc(
            gamma_m, q_dist, m_p, clust_synth, obs_uncert, cluster_lkl,
            ran_x, False)
        if lkl > lkl_old:
            gamma_m_opt = gamma_m
            lkl_old = lkl
        # bar()

    IMF_lkl_params = gamma_m_opt
    IMF_synth_clusts = minFunc(
        gamma_m_opt, q_dist, m_p, clust_synth, obs_uncert, cluster_lkl, ran_x,
        True)

    return lkl_old, IMF_lkl_params, IMF_synth_clusts


# def diffEvol(q_dist, cluster_lkl, obs_uncert, clust_synth, masses_interp):
#     """
#     """
#     # Convert masses to [0, 1] range
#     m_p = (masses_interp - 0.08) / (150 - 0.08)

#     bounds = [(gamma_m_min, gamma_m_max)]
#     res = DE(minFunc, bounds, tol=DE_tol, args=(
#         q_dist, m_p, clust_synth, obs_uncert, cluster_lkl, False, 'DE'))

#     IMF_lkl_params = res.x[0]
#     IMF_synth_clusts = minFunc(
#         IMF_lkl_params, q_dist, m_p, clust_synth, obs_uncert, cluster_lkl,
#         True, 'DE')

#     return res.fun, IMF_lkl_params, IMF_synth_clusts


def gammaFunc(x, gamma_m):
    """
    Power-law must be normalized so that the 'gamma_p' values can be
    converted to the 'gamma_q' range
    """
    y = gamma_m * x**(gamma_m - 1)
    y /= y.max()
    return y


def InvgammaFunc(m_p, gamma_m, ran_x):
    """
    """
    # Using 'gamma_m', obtain the 'gamma_p' values for each (normalized) mass
    gamma_p = gammaFunc(m_p, gamma_m)

    # Convert 'gamma_p' to the selected 'gamma_q' range
    gamma_q = gamma_q_min + (gamma_q_max - gamma_q_min) * gamma_p

    # Sample the 'q' values used for the masses
    # x = np.random.uniform(0, 1, len(gamma_q))
    q_vals = ran_x**(1 / gamma_q)

    return q_vals


def minFunc(
    gamma_m, q_dist, m_p, clust_synth, obs_uncert, cluster_lkl, ran_x,
        ret_clust, BF_DE='BF'):
    """
    """
    # Using the 'gamma_p' values obtain the 'q' values for each mass
    q_vals = InvgammaFunc(m_p, gamma_m, ran_x)

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
