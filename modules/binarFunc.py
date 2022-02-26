
import numpy as np
# from scipy.stats import powerlaw
# from scipy.spatial.distance import cdist


# Mass range for single stars
# mmin, mmax = 0.08, 150


def generate(q_dist, isoch_phot, mass_ini, isoch_col_mags):
    """
    """
    isoch_binaries = []
    for q in q_dist:
        m2 = q * mass_ini
        i2 = np.searchsorted(mass_ini, m2)
        Gmag_s1, Gmag_s2 = isoch_phot[0], isoch_phot[0, i2]
        BPmag_1, RPmag_1 = isoch_col_mags[0], isoch_col_mags[1]
        BPmag_2, RPmag_2 = isoch_col_mags[0, i2], isoch_col_mags[1, i2]

        mag_binar = mag_combine(Gmag_s1, Gmag_s2)
        m1_col_binar = mag_combine(BPmag_1, BPmag_2)
        m2_col_binar = mag_combine(RPmag_1, RPmag_2)
        col_binar = m1_col_binar - m2_col_binar

        # import matplotlib.pyplot as plt
        # plt.scatter(isoch_phot[1], isoch_phot[0])
        # plt.scatter(col_binar, mag_binar, alpha=.5, zorder=2, c='r')
        # plt.gca().invert_yaxis()
        # plt.show()

        isoch_binaries.append([mag_binar, col_binar])

    return np.array(isoch_binaries)


def mag_combine(m1, m2):
    """
    Combine two magnitudes. This is a faster re-ordering of the standard
    formula:

    -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))

    """

    # 10**-.4 = 0.398107
    mbin = -2.5 * (-.4 * m1 + np.log10(1. + 0.398107 ** (m2 - m1)))

    return mbin


def QDistFunc(gamma_q, M1_masses):
    """
    This function maps the primary masses 'M1' with their corresponding 'q'
    value (mass-ratio), later used to select the secondary mass.
    """

    # q_vals = gamma_q * M1_masses ** (gamma_q - 1)
    # q_vals = np.clip(q_vals, a_min=0, a_max=1)

    # def CDF(x, g):
    #     """
    #     """
    #     return (x**g - qmin**g) / (qmax**g - qmin**g)

    # def CDF_inv(x, g):
    #     """
    #     """
    #     return ((qmax**g - qmin**g) * x + qmin**g) ** (1 / g)

    x = np.random.uniform(0, 1, len(M1_masses))
    q_vals = x**(1 / gamma_q)

    # # M2_masses = M1_masses * q_vals
    # # M12_masses = M1_masses + M2_masses
    # import matplotlib.pyplot as plt
    # plt.subplot(131)
    # plt.hist(gamma_q)
    # plt.xlabel("gamma_q")
    # plt.subplot(132)
    # plt.hist(q_vals)
    # plt.xlabel("q_vals")
    # plt.subplot(133)
    # plt.scatter(M1_masses, q_vals)
    # plt.xlabel("M1")
    # plt.ylabel("gamma_q")
    # plt.show()

    return q_vals


def addBinaries(q_dist, clust_synth, q_vals):
    """
    """

    idx = closestIdx(q_dist, q_vals, len(q_vals))

    # arr = []
    # for i, st in enumerate(clust_synth.T):
    #     arr.append(st[:, idx[i]])
    # single_systs = np.array(arr).T

    # Source: https://stackoverflow.com/a/71268132/1391441
    binar_systs = clust_synth[idx, :, np.arange(clust_synth.shape[2])].T

    # import matplotlib.pyplot as plt
    # plt.scatter(clust_synth[0][1], clust_synth[0]
    #             [0], marker='x', c='k', zorder=0)
    # plt.scatter(arr[1], arr[0], c='r', alpha=.5)
    # plt.gca().invert_yaxis()
    # plt.show()
    # breakpoint()

    return binar_systs


def closestIdx(arr1, arr2, N):
    """
    Source: https://stackoverflow.com/a/21391265/1391441
    """

    def find_closest(A, target):
        # A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    order = arr1.argsort()
    key = arr1[order]
    target = arr2[:N]
    closest = find_closest(key, target)

    return order[closest]
