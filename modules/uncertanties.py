
import numpy as np
from scipy.optimize import curve_fit


def getUncert(cluster):
    '''
    Fit an exponential function to the errors in each photometric dimension,
    using the main magnitude as the x coordinate.
    This data is used to display the error bars, and more importantly, to
    generate the synthetic clusters in the best match module.
    '''
    # Use the main magnitude after max error rejection.
    mmag = cluster[0]
    be_m, interv_mag, n_interv, mmag_interv_pts = errorData(mmag)

    # Obtain the median points for photometric errors. Append magnitude
    # values first, and colors after.
    e_mag, e_col = cluster[2], cluster[3]

    e_mc_medians = []
    for e_mc in [e_mag, e_col]:
        e_mc_medians.append(err_medians(
            mmag, e_mc, be_m, interv_mag, n_interv))

    # Fit exponential curve for each photometric error dimension.
    err_lst = []
    for e_mc_v in e_mc_medians:
        popt_mc = get_m_c_errors(mmag, e_mc_v, mmag_interv_pts)
        err_lst.append(popt_mc)

    return err_lst


def errorData(mmag):
    """
    Use the main magnitude to determine the error parameters that will be
    used in the synthetic cluster generation.
    """
    # Define 'bright end' leaving out the brightest stars.
    m_s = np.sort(mmag)
    # Magnitude of the .5% brightest star
    m_i = int(m_s.size * 0.005)
    be_m = max(min(mmag) + 1., m_s[m_i])
    # Magnitude range.
    delta_mag = max(mmag) - be_m
    # Width of the intervals in magnitude.
    interv_mag = 0.5
    # Number of intervals.
    n_interv = int(round(delta_mag / interv_mag))
    # Create a segmented list in magnitude.
    # Define list of points spanning the magnitude range starting from the
    # bright end. The '+ interv_mag' term is intentional so that the
    # err_medians function defines ranges around these values and they get
    # positioned in the middle of the magnitude interval.
    if n_interv < 2:
        print("  WARNING: main magnitude range is very small")
    mmag_interv_pts = [
        be_m + interv_mag * (q + interv_mag) for q in range(n_interv)]

    return be_m, interv_mag, n_interv, mmag_interv_pts


def err_medians(mmag, e_mc, be_m, interv_mag, n_interv):
    '''
    Store median of photometric errors for each main magnitude interval
    Do this for magnitude and color errors.
    '''
    # Each list within the 'mc_interv' list holds all the photometric error
    # values for all the stars in the interval 'q' which corresponds to the
    # mag value:
    # [bright_end+(interv_mag*q) + bright_end+(interv_mag*(q+1))]/2.
    # where 'q' is the index that points to the interval being filled.
    mc_interv = [[] for _ in range(n_interv)]

    # Iterate through all stars.
    for st_ind, st_mag in enumerate(mmag):

        # Use only stars above the bright end. All stars are already below
        # the err_max limit.
        if be_m <= st_mag:
            # Store each star in its corresponding interval in the segmented
            # mag list. Will be used to calculate the curve fit.

            # Iterate through all intervals in magnitude.
            for q in range(n_interv):
                # Store star's errors in corresponding interval.
                if (be_m + interv_mag * q) <= st_mag < (be_m + interv_mag
                                                        * (q + 1)):
                    # Star falls in this interval, store its error value.
                    mc_interv[q].append(e_mc[st_ind])
                    break

    # We have the photometric errors of stars within the (be_m, err_max) range,
    # stored in magnitude intervals (from the main magnitude) in the
    # 'mc_interv' list.

    # 'e_mc_value' will hold the median photometric error for each interval
    # of the main magnitude.
    e_mc_value = []
    # Initial value for the median.
    median = 0.0001
    # Iterate through all intervals (lists) in the main magnitude range.
    for interv in mc_interv:
        # Check that list is not empty.
        if interv:
            median = np.median(interv)
        e_mc_value.append(median)

    return e_mc_value


def get_m_c_errors(mags, e_mc_v, mmag_interv_pts):
    '''
    Fit 3P or 2P exponential curve.
    '''
    try:
        if len(mmag_interv_pts) >= 3:
            # Fit 3-param exponential curve.
            popt_mc, dummy = curve_fit(exp_3p, mmag_interv_pts, e_mc_v)
        else:
            # If the length of this list is 2, it means that the main
            # magnitude length is too small. If this is the case, do not
            # attempt to fit a 3 parameter exp function since it will fail.
            raise RuntimeError

    # If the 3-param exponential fitting process fails.
    except RuntimeError:
        print("  3P exponential error function fit failed. Attempt 2P fit")
        try:
            # Fit simple 2-params exponential curve.
            popt_mc, dummy = curve_fit(exp_2p, mmag_interv_pts, e_mc_v)
            # Insert empty 'c' value to be fed later on to the 3P exponential
            # function used to obtain the plotted error bars. This makes the
            # 2P exp function equivalent with the 3P exp function, with the
            # 'c' parameter equal to 0.
            popt_mc = np.insert(popt_mc, 2, 0.)

        # If the 2-param exponential fitting process also fails, try with a
        # 2P exp but using only min and max error values.
        except RuntimeError:
            print("  2P exponential error function fit failed"
                  " Perform min-max magnitude fit.")
            # Fit simple 2-params exponential curve.
            mmag_interv_pts = [
                min(mags), max(mags) - (max(mags) - min(mags)) / 20.]
            e_mc_r = [min(e_mc_v), max(e_mc_v)]
            popt_mc, dummy = curve_fit(exp_2p, mmag_interv_pts, e_mc_r)
            # Insert 'c' value into exponential function param list.
            popt_mc = np.insert(popt_mc, 2, 0.)

    return popt_mc


def addErrors(cl_synth, err_lst):
    """
    Add random synthetic uncertainties to the magnitude and color(s)
    """

    rnd = np.random.normal(0., 1., cl_synth.shape[-1])

    # import matplotlib.pyplot as plt
    # plt.scatter(cl_synth[1], cl_synth[0], c='g', alpha=.5)

    for i, popt_mc in enumerate(err_lst):
        # cl_synth[0] is the main magnitude.
        sigma_mc = getSigmas(cl_synth[0], popt_mc)

        # Randomly move stars around these errors.
        cl_synth[i] = gauss_error(rnd, cl_synth[i], sigma_mc)

    # plt.scatter(cl_synth[1], cl_synth[0], c='r', alpha=.5)
    # plt.gca().invert_yaxis()
    # plt.show()

    return cl_synth


def getSigmas(main_mag, popt_mc):
    """
    Uncertainties for each photometric dimension
    """
    return exp_3p(main_mag, *popt_mc)


def exp_3p(x, a, b, c):
    """
    Three-parameters exponential function.

    This function is tied to the 'synth_cluster.add_errors' function.
    """
    return a * np.exp(b * x) + c


def exp_2p(x, a, b):
    """
    Two-parameters exponential function.
    """
    return a * np.exp(b * x)


def gauss_error(rnd, mc, e_mc):
    """
    Randomly move mag and color through a Gaussian function.

    mc  : magnitude or color dimension
    rnd : random array of floats normally distributed around 0. with stddev 1.
    e_mc: fitted observational uncertainty value
    """
    mc_gauss = mc + rnd[:len(mc)] * e_mc

    # import matplotlib.pyplot as plt
    # plt.hist(rnd[:len(mc)] * e_mc)
    # plt.show()

    return mc_gauss
