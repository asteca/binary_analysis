
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from astropy.io import ascii
from scipy.optimize import differential_evolution as DE
from modules import readData, uncertanties, isochHandle, binarFunc, imfFunc,\
    massInterp, likelihood
from modules.HARDCODED import cmd_systs, idx_header
import time as t
from pyinstrument import Profiler


# Array for the q(=M2/M1) mass-ratio values. This determines how many
# isochrones will be stored in 'isoch_binaries'. A larger number means
# better resolution (at the expense of more memory used)
q_min, q_max = 0.01, .99
q_dist = np.linspace(q_min, q_max, 500)

# gamma range used to find the gamma_q values that are used to sample the q
gamma_m_min, gamma_m_max = .1, 10

gamma_q_min, gamma_q_max = .1, 10

bf_cut = .2
DE_tol = 0.001

# Number of runs with a new IMF re-sample
N_IMF = 500


def main(profiler):
    """
    """
    cluster_params = ascii.read("clust_params.dat")
    # Table  to dictionary
    cl_dict = {}
    for cl in cluster_params:
        cl_dict[cl['cluster']] = (
            cl['metal'], cl['age'], cl['e_bv'], cl['dist_mod'])

    clusters = readFiles()
    for cluster_path in clusters:
        cl_name = cluster_path.name[:-4]
        params = cl_dict[cl_name]

        print("Load cluster data")
        cluster = readData.loadClust(cl_name)
        mag_max = cluster[0].max()

        cluster_lkl = likelihood.prepCluster(cluster)

        obs_uncert = uncertanties.getUncert(cluster)

        print("Load isochrone data")
        turn_off, isoch_phot, mass_ini, isoch_col_mags =\
            isochHandle.isochProcess(cmd_systs, idx_header, mag_max, *params)

        # plt.scatter(cluster[1], cluster[0], alpha=.5)
        # plt.scatter(isoch_phot[1], isoch_phot[0], alpha=.5, c='r')
        # plt.gca().invert_yaxis()
        # plt.show()

        print("Generate binary systems for all 'q'")
        isoch_binaries = binarFunc.generate(
            q_dist, isoch_phot, mass_ini, isoch_col_mags)

        # plt.scatter(cluster[1], cluster[0], alpha=.5)
        # plt.scatter(
        #     isoch_binaries[0][1], isoch_binaries[0][0], alpha=.5, c='r')
        # plt.gca().invert_yaxis()
        # plt.show()

        N_stars = len(cluster[0])

        # Identify the mass range in the observed cluster.
        mag_max = cluster[0].max()
        # mag_min = cluster[0].min()
        mass_min_i = np.argmin(abs(mag_max - isoch_phot[0]))
        mass_min = mass_ini[mass_min_i]
        # mass_max_i = np.argmin(abs(mag_min - isoch_phot[0]))
        # mass_max = mass_ini[mass_max_i]

        # print("Generate binary prob samples")
        # GFunc_samples = binarFunc.GammaFunction(
        #     gamma_m_lst, N_stars, mass_min, mass_max)

        print("Generate IMF")
        IMF_inv_cdf = imfFunc.IMF_CDF()

        lkl_lst = []
        for Ni in range(N_IMF):
            # print(Ni)
            # profiler.start()

            # Sample the IMF
            masses = imfFunc.IMFSample(N_stars, mass_min, IMF_inv_cdf)

            # Interpolate masses from the IMF into the isochrone
            clust_synth, masses_interp = massInterp.interp(
                isoch_binaries, mass_ini, masses)

            # # For each mass in 'masses_interp' assign a number in the [0, 1]
            # # range distributed interwined
            # # TODO just use a random.uniform() distribution?
            # N_M1 = len(masses_interp)
            # N1 = int(len(masses_interp) * .5)
            # N2 = len(masses_interp) - N1
            # if N_M1 % 2 == 1:
            #     N2 -= 1
            # x1 = np.linspace(0, 1, N1)
            # x2 = np.linspace(1, 0, N2)
            # # Intertwine
            # x = np.empty((N1 + N2))
            # x[0::2] = x1
            # x[1::2] = x2
            # if N_M1 % 2 == 1:
            #     x = np.append(x, [.5])
            # masses_interwined = x
            # # masses_interwined = np.random.uniform(0, 1, len(masses_interp))

            # IMF_lkl_params, IMF_synth_clusts = diffEvol(
            #     cluster_lkl, obs_uncert, clust_synth, masses_interp)

            IMF_lkl_params, IMF_synth_clusts = bruteForce(
                cluster_lkl, obs_uncert, clust_synth, masses_interp)

            # profiler.stop()
            # print("Preparing profiler report...")
            # profiler.open_in_browser()

            q_vals, sc = IMF_synth_clusts
            b_msk = q_vals > bf_cut

            s_msk = ~b_msk
            plt.subplot(221)
            plt.scatter(cluster[1], cluster[0], alpha=.5)
            plt.scatter(sc[1][s_msk], sc[0][s_msk], alpha=.5, c='g')
            plt.scatter(sc[1][b_msk], sc[0][b_msk], alpha=.5, c='r')
            plt.gca().invert_yaxis()
            plt.subplot(222)
            plt.hist(q_vals[b_msk])
            # plt.axvline(.2, c='r', zorder=2)
            M1_binr_masses = masses_interp[b_msk]
            M1_sing_masses = masses_interp[s_msk]
            Mb_h, edges = np.histogram(M1_binr_masses, 5)
            Ms_h, _ = np.histogram(M1_sing_masses, edges)
            Mt_h = Ms_h + Mb_h
            x = .5 * (edges[1:] + edges[:-1])
            plt.subplot(223)
            plt.scatter(x, Mb_h / Mt_h)
            plt.show()
            print(x)
            print(Mb_h / Mt_h)
            breakpoint()

            bf = b_msk.sum() / masses_interp.size
            lkl_lst.append([IMF_lkl_params, bf])

        lkl_lst = np.array(lkl_lst).T
        plt.subplot(131)
        plt.scatter(*lkl_lst)
        plt.subplot(132)
        plt.hist(lkl_lst[0])
        plt.subplot(133)
        plt.hist(lkl_lst[1])
        plt.show()
        breakpoint()


def bruteForce(cluster_lkl, obs_uncert, clust_synth, masses_interp):
    """
    """
    gamma_m_lst = np.linspace(gamma_m_min, gamma_m_max, 100)

    # Convert masses to [0, 1] range
    m_p = (masses_interp - 0.08) / (150 - 0.08)

    lkl_old, gamma_m_opt = 0, 1000
    # with alive_bar(100 * 1000) as bar:
    for gamma_m in gamma_m_lst:
        for _ in range(100):
            lkl = minFunc(
                gamma_m, m_p, clust_synth, obs_uncert, cluster_lkl, False)
            if lkl > lkl_old:
                gamma_m_opt = gamma_m
                lkl_old = lkl
        # bar()

    IMF_lkl_params = gamma_m_opt
    IMF_synth_clusts = minFunc(
        gamma_m_opt, m_p, clust_synth, obs_uncert, cluster_lkl, True)

    b_msk = IMF_synth_clusts[0] > bf_cut
    bf = b_msk.sum() / masses_interp.size

    print("BF | L={:.2f},  bf={:.2f}, gamma_m={:.2f},".format(
        lkl_old, bf, gamma_m_opt))

    return IMF_lkl_params, IMF_synth_clusts


def diffEvol(cluster_lkl, obs_uncert, clust_synth, masses_interp):
    """
    """
    # Convert masses to [0, 1] range
    m_p = (masses_interp - 0.08) / (150 - 0.08)

    # bounds = [(gamma_m_min, gamma_m_max), (q_min, q_max), (.1, 3)]
    bounds = [(gamma_m_min, gamma_m_max)]
    res = DE(minFunc, bounds, tol=DE_tol, args=(
        m_p, clust_synth, obs_uncert, cluster_lkl, False))

    IMF_lkl_params = res.x[0]
    IMF_synth_clusts = minFunc(
        IMF_lkl_params, m_p, clust_synth, obs_uncert, cluster_lkl, True)

    b_msk = IMF_synth_clusts[0] > bf_cut
    bf = b_msk.sum() / masses_interp.size

    print("DE | L={:.2f},  bf={:.2f}, gamma_m={:.2f},".format(
        res.fun, bf, *res.x))

    return IMF_lkl_params, IMF_synth_clusts


def gammaFunc(x, gamma_m):
    """
    Power-law must be normalized so that the 'gamma_p' values can be
    converted to the 'gamma_q' range
    """
    y = gamma_m * x**(gamma_m - 1)
    y /= y.max()
    return y


def InvgammaFunc(gamma_p):
    """
    """
    # Convert 'gamma_p' to the selected 'gamma_q' range
    gamma_q = gamma_q_min + (gamma_q_max - gamma_q_min) * gamma_p

    # Sample the 'q' values used for the masses
    x = np.random.uniform(0, 1, len(gamma_q))
    q_vals = x**(1 / gamma_q)

    return q_vals


def minFunc(
        gamma_m, m_p, clust_synth, obs_uncert, cluster_lkl, ret_clust):
    """
    """
    # Using 'gamma_m', obtain the 'gamma_p' values for each (normalized) mass
    gamma_p = gammaFunc(m_p, gamma_m)

    # Using the 'gamma_p' values obtain the 'q' values for each mass
    q_vals = InvgammaFunc(gamma_p)

    # Are q_vals being generated such that M2 < 0.08?

    clust_synth_binar = binarFunc.addBinaries(
        q_dist, clust_synth, q_vals)

    clust_synth_final = uncertanties.addErrors(
        clust_synth_binar, obs_uncert)

    if ret_clust:
        return q_vals, clust_synth_final

    Lkl = likelihood.getLkl(cluster_lkl, clust_synth_final)

    # return -Lkl
    return Lkl


def readFiles(in_folder='input'):
    """
    Read files from the input folder
    """
    files = []
    for pp in Path(in_folder).iterdir():
        if pp.is_file():
            files += [pp]
        # else:
        #     files += [arch for arch in pp.iterdir()]

    return files


if __name__ == '__main__':
    profiler = Profiler(interval=0.0001)
    main(profiler)
