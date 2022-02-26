
import numpy as np
from astropy.stats import knuth_bin_width
from scipy.special import loggamma
from fast_histogram import histogram2d


def prepCluster(cluster):
    """
    """

    # mags_cols_cl = dataProcess(cluster)
    cl_phot = cluster[:2]

    # Obtain bin edges for each dimension, defining a grid.
    bin_edges = bin_edges_f(cl_phot)

    # Obtain histogram for observed cluster.
    cl_histo = np.histogramdd(cl_phot.T, bins=bin_edges)[0]

    # Flatten N-dimensional histograms.
    cl_histo_f = cl_histo.ravel()

    # Index of bins where stars were observed.
    cl_z_idx = (cl_histo_f != 0)

    # Remove all bins where n_i=0 (no observed stars)
    cl_histo_f_z = cl_histo_f[cl_z_idx]

    x0, x1 = bin_edges[0].min(), bin_edges[0].max()
    y0, y1 = bin_edges[1].min(), bin_edges[1].max()
    nx, ny = len(bin_edges[0]) - 1, len(bin_edges[1]) - 1
    bin_edges = (x0, x1, y0, y1, nx, ny)

    obs_clust = (bin_edges, cl_histo_f_z, cl_z_idx)

    return obs_clust


def bin_edges_f(cl_phot, bin_method='knuth', min_bins=2, max_bins=50):
    """
    Obtain bin edges for each photometric dimension using the cluster region
    diagram. The 'bin_edges' list will contain all magnitudes first, and then
    all colors (in the same order in which they are read).
    """

    bin_edges = []
    if bin_method in (
            'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'):

        bin_edges.append(np.histogram(cl_phot[0], bins=bin_method)[1])
        bin_edges.append(np.histogram(cl_phot[1], bins=bin_method)[1])

    elif bin_method == 'fixed':
        # Based on Bonatto & Bica (2007) 377, 3, 1301-1323 but using larger
        # values than those used there (0.25 for colors and 0.5 for magnitudes)
        b_num = int(round(max(2, (max(cl_phot[0]) - min(cl_phot[0])) / 1.)))
        bin_edges.append(np.histogram(cl_phot[0], bins=b_num)[1])
        b_num = int(round(max(2, (max(
            cl_phot[1]) - min(cl_phot[1])) / .5)))
        bin_edges.append(np.histogram(cl_phot[1], bins=b_num)[1])

    elif bin_method == 'knuth':
        bin_edges.append(knuth_bin_width(
            cl_phot[0], return_bins=True, quiet=True)[1])
        bin_edges.append(knuth_bin_width(
            cl_phot[1], return_bins=True, quiet=True)[1])

    # Impose a minimum of 'min_bins' cells per dimension. The number of bins
    # is the number of edges minus 1.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins < min_bins:
            # print("  WARNING too few bins in histogram, use 'min_bins'")
            bin_edges[i] = np.linspace(be[0], be[-1], min_bins + 1)

    # Impose a maximum of 'max_bins' cells per dimension.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins > max_bins:
            # print("  WARNING too many bins in histogram, use 'max_bins'")
            bin_edges[i] = np.linspace(be[0], be[-1], max_bins)

    return bin_edges


def getLkl(obs_clust, synth_clust):
    """
    Match the synthetic cluster to the observed cluster.
    """

    # # If synthetic cluster is empty, assign a small likelihood value.
    # if not synth_clust.any():
    #     return -1.e09

    # Observed cluster's data.
    bin_edges, cl_histo_f_z, cl_z_idx = obs_clust

    # Histogram of the synthetic cluster, using the bin edges calculated
    # with the observed cluster.
    x0, x1, y0, y1, nx, ny = bin_edges
    syn_histo = histogram2d(
        synth_clust[0], synth_clust[1], range=[[x0, x1], [y0, y1]],
        bins=(nx, ny))

    # Flatten N-dimensional histogram.
    syn_histo_f = syn_histo.ravel()
    # Remove all bins where n_i = 0 (no observed stars).
    syn_histo_f_z = syn_histo_f[cl_z_idx]

    SumLogGamma = np.sum(
        loggamma(cl_histo_f_z + syn_histo_f_z + .5)
        - loggamma(syn_histo_f_z + .5))

    # M = synth_clust.shape[0]
    # ln(2) ~ 0.693
    tremmel_lkl = SumLogGamma - 0.693 * synth_clust.shape[0]

    return tremmel_lkl
