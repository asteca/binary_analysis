
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


# Low mass limits for each IMF.
m_low, m_high = 0.08, 150


def IMFSample(N_stars, mass_min, IMF_inv_cdf):
    """
    """
    def sampled_inv_cdf(N):
        mr = np.random.rand(N)
        return IMF_inv_cdf(mr)

    # Sample in chunks of 100 until the required number of stars is reached.
    # Reject stars below the minimum observed mass 'mass_min'
    mass_samples = []
    while len(mass_samples) < N_stars:
        masses = sampled_inv_cdf(100)
        msk = masses > mass_min
        mass_samples += list(masses[msk])

    return np.array(mass_samples)[:N_stars]


def IMF_CDF(mass_step=0.05):
    """
    Generate the inverse CDF for the selected IMF.
    """

    # IMF mass interpolation step and grid values.
    mass_values = np.arange(m_low, m_high, mass_step)

    # The CDF is defined as: $F(m)= \int_{m_low}^{m} PDF(m) dm$
    # Sample the CDF
    CDF_samples = []
    for m in mass_values:
        CDF_samples.append(quad(imfs, m_low, m)[0])
    CDF_samples = np.array(CDF_samples)

    # Normalize CDF
    CDF_samples /= CDF_samples.max()
    # Inverse CDF
    inv_cdf = interp1d(CDF_samples, mass_values)

    return inv_cdf


def imfs(m_star, IMF_name="kroupa_2002", mass_flag=False):
    """
    Define any number of IMFs.

    The package https://github.com/keflavich/imf has some more (I think,
    24-09-2019).
    """

    if IMF_name == 'kroupa_1993':
        # Kroupa, Tout & Gilmore. (1993) piecewise IMF.
        # http://adsabs.harvard.edu/abs/1993MNRAS.262..545K
        # Eq. (13), p. 572 (28)
        alpha = [-1.3, -2.2, -2.7]
        m0, m1, m2 = [0.08, 0.5, 1.]
        factor = [0.035, 0.019, 0.019]
        if m0 < m_star <= m1:
            i = 0
        elif m1 < m_star <= m2:
            i = 1
        elif m2 < m_star:
            i = 2
        imf_val = factor[i] * (m_star ** alpha[i])

    elif IMF_name == 'kroupa_2002':
        # Kroupa (2002) Salpeter (1995) piecewise IMF taken from MASSCLEAN
        # article, Eq. (2) & (3), p. 1725
        alpha = [-0.3, -1.3, -2.3]
        m0, m1, m2 = [0.01, 0.08, 0.5]
        factor = [(1. / m1) ** alpha[0], (1. / m1) ** alpha[1],
                  ((m2 / m1) ** alpha[1]) * ((1. / m2) ** alpha[2])]
        if m0 <= m_star <= m1:
            i = 0
        elif m1 < m_star <= m2:
            i = 1
        elif m2 < m_star:
            i = 2
        imf_val = factor[i] * (m_star ** alpha[i])

    elif IMF_name == 'chabrier_2001_log':
        # Chabrier (2001) lognormal form of the IMF.
        # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
        # Eq (7)
        imf_val = (1. / (np.log(10) * m_star)) * 0.141 * \
            np.exp(-((np.log10(m_star) - np.log10(0.1)) ** 2)
                   / (2 * 0.627 ** 2))

    elif IMF_name == 'chabrier_2001_exp':
        # Chabrier (2001) exponential form of the IMF.
        # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
        # Eq (8)
        imf_val = 3. * m_star ** (-3.3) * np.exp(-(716.4 / m_star) ** 0.25)

    elif IMF_name == 'salpeter_1955':
        # Salpeter (1955)  IMF.
        # https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/
        imf_val = m_star ** -2.35

    if mass_flag:
        imf_val *= m_star

    return imf_val
