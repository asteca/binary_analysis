
import numpy as np
# from scipy.stats import powerlaw
# from scipy.spatial.distance import cdist


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


def addBinaries(q_dist, clust_synth, q_vals):
    """
    """

    idx = closestIdx(q_dist, q_vals)

    # # 'IndexError: shape mismatch'. Apparently one star goes missing in 'idx'
    # if len(idx) < clust_synth.shape[2]:
    #     idx = np.array(list(idx) + [0])

    # Source: https://stackoverflow.com/a/71268132/1391441
    # When using the DE sometimes this will fail with
    try:
        binar_systs = clust_synth[idx, :, np.arange(clust_synth.shape[2])].T
    except IndexError:
        print("error")
        breakpoint()

    # import matplotlib.pyplot as plt
    # plt.scatter(clust_synth[0][1], clust_synth[0]
    #             [0], marker='x', c='k', zorder=0)
    # plt.scatter(arr[1], arr[0], c='r', alpha=.5)
    # plt.gca().invert_yaxis()
    # plt.show()
    # breakpoint()

    return binar_systs


def closestIdx(A, target):
    """
    Source: https://stackoverflow.com/a/21391265/1391441

    arr1 (or A) is already ordered, so we skip the rest
    """
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

# def closestIdx(arr1, arr2):
#     """
#     Source: https://stackoverflow.com/a/21391265/1391441
#     """

#     def find_closest(A, target):
#         # A must be sorted
#         idx = A.searchsorted(target)
#         idx = np.clip(idx, 1, len(A) - 1)
#         left = A[idx - 1]
#         right = A[idx]
#         idx -= target - left < right - target
#         return idx

#     # 'arr1' is already sorted, so we can skip this
#     # order = arr1.argsort()
#     # key = arr1[order]
#     # target = arr2
#     # closest = find_closest(key, target)
#     # return order[closest]

#     return find_closest(arr1, arr2)
