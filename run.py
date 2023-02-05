
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from modules import readData, uncertanties, isochHandle, binarFunc, imfFunc,\
    likelihood, massInterp, optMethod
from modules.HARDCODED import cmd_systs, idx_header, q_min, q_max, q_dist_N,\
    N_IMF, bf_cut

# from pyinstrument import Profiler


def main(profiler):
    """
    """
    q_dist = np.linspace(q_min, q_max, q_dist_N)

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

        # TODO temp, remove stars that look bad
        msk = cluster[1] < 2.62
        cluster = cluster[:, msk]

        cluster_lkl = likelihood.prepCluster(cluster)

        obs_uncert = uncertanties.getUncert(cluster)

        print("Load isochrone data")
        turn_off, isoch_phot, mass_ini, isoch_col_mags =\
            isochHandle.isochProcess(cmd_systs, idx_header, *params)

        print("Generate binary systems for all 'q'")
        isoch_binaries = binarFunc.generate(
            q_dist, isoch_phot, mass_ini, isoch_col_mags)

        N_stars = len(cluster[0])

        # Identify the mass range in the observed cluster.
        mag_max = cluster[0].max()
        # mag_min = cluster[0].min()
        mass_min_i = np.argmin(abs(mag_max - isoch_phot[0]))
        mass_min = mass_ini[mass_min_i]
        # mass_max_i = np.argmin(abs(mag_min - isoch_phot[0]))
        # mass_max = mass_ini[mass_max_i]
        # mass_intervs = np.linspace(mass_min - 0.001, mass_max + 0.001, 5)

        print("Generate IMF")
        IMF_inv_cdf = imfFunc.IMF_CDF()

        lkl_lst = []
        for Ni in range(N_IMF):
            # profiler.start()

            # Sample the IMF
            masses = imfFunc.IMFSample(N_stars, mass_min, IMF_inv_cdf)

            # Interpolate masses from the IMF into the isochrone
            clust_synth, masses_interp = massInterp.interp(
                isoch_binaries, mass_ini, masses)

            # # Order the synthetic clusters by mass (min, max)
            # idx = np.argsort(masses_interp)
            # masses_interp = masses_interp[idx]
            # clust_synth = clust_synth[:, :, idx]

            plt.suptitle("BF results")

            print("")

            # print("Method 1")
            # lkl, IMF_lkl_params, IMF_synth_clusts =\
            #     method1.bruteForce(
            #         q_dist, cluster_lkl, obs_uncert, clust_synth,
            #         masses_interp)
            # plotMethods(
            #     1, cluster, masses_interp, lkl, IMF_lkl_params,
            #     IMF_synth_clusts, 4)

            mass_intervs = np.linspace(
                masses_interp.min() - 0.001, masses_interp.max() + 0.001, 5)

            masses_mask = []
            i_old = 0
            for mass_interv_f in mass_intervs[1:]:
                msk = (mass_intervs[i_old] < masses_interp) &\
                    (masses_interp <= mass_interv_f)
                masses_mask.append(msk)
                i_old += 1

            #
            # "M0" "M2" "M3" "M4", "M7",
            for i, MM in enumerate(("M1", "M8")):
                lkl, IMF_lkl_params, IMF_synth_clusts =\
                    optMethod.bruteForce(
                        MM, q_dist, cluster_lkl, obs_uncert, clust_synth,
                        masses_mask)
                #
                q_vals, sc = IMF_synth_clusts
                b_msk = q_vals > bf_cut
                bf = b_msk.sum() / masses_interp.size
                print("{} | L={:.2f},  bf={:.2f}, gamma_q=".format(
                    MM, lkl, bf), np.round(IMF_lkl_params, 2))

                plotMethods(
                    i + 1, MM, cluster, masses_interp, lkl, IMF_lkl_params,
                    IMF_synth_clusts, mass_intervs)

            plt.show()

            # profiler.stop()
            # print("Preparing profiler report...")
            # profiler.open_in_browser()

            # plt.show()

        #     lkl_lst.append([IMF_lkl_params, bf])

        # lkl_lst = np.array(lkl_lst).T
        # plt.subplot(131)
        # plt.scatter(*lkl_lst)
        # plt.subplot(132)
        # plt.hist(lkl_lst[0])
        # plt.subplot(133)
        # plt.hist(lkl_lst[1])
        # plt.show()
        # breakpoint()


def plotMethods(
    i, MM, cluster, masses_interp, lkl, IMF_lkl_params, IMF_synth_clusts,
        mass_intervs):
    """
    """
    q_vals, sc = IMF_synth_clusts
    b_msk = q_vals > bf_cut

    if i == 1:
        pi = ("1", "2", "3", "4")
    elif i == 2:
        pi = ("5", "6", "7", "8")

    s_msk = ~b_msk
    plt.subplot(int("24" + pi[0]))
    plt.title("Lkl: {:.2f}".format(lkl))
    plt.scatter(cluster[1], cluster[0], alpha=.5)
    try:
        plt.scatter(sc[1][s_msk], sc[0][s_msk], alpha=.5, c='g')
        plt.scatter(sc[1][b_msk], sc[0][b_msk], alpha=.5, c='r')
    except IndexError:
        print("error 2")
        breakpoint()
    plt.gca().invert_yaxis()

    plt.subplot(int("24" + pi[1]))
    plt.hist(q_vals[b_msk], density=True)

    M1_binr_masses = masses_interp[b_msk]
    M1_sing_masses = masses_interp[s_msk]
    Mb_h, edges = np.histogram(M1_binr_masses, mass_intervs)
    Ms_h, _ = np.histogram(M1_sing_masses, edges)
    Mt_h = Ms_h + Mb_h
    x = .5 * (edges[1:] + edges[:-1])
    plt.subplot(int("24" + pi[2]))
    plt.scatter(x, Mb_h / Mt_h)
    plt.scatter((.3, 1, 2.42, 7.75), (.248, .445, .5, 0.6),
                marker='s', c='r')
    plt.xlabel("M2/M1")
    plt.ylim(0, 1)

    plt.subplot(int("24" + pi[3]))
    plt.scatter(masses_interp, q_vals)
    # breakpoint()
    # plt.boxplot()


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
    main(None)
    # profiler = Profiler(interval=0.0001)
    # main(profiler)
