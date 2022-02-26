
import numpy as np


def interp(isoch_binars, mass_ini, masses):
    """
    For each mass in the sampled IMF mass distribution, find the star in the
    isochrone  with the closest initial mass value and pass it forward.
    Masses that fall outside of the isochrone's mass range are rejected.
    """
    # Reject masses in the IMF mass distribution that are located outside of
    # the theoretical isochrone's mass range.
    masses = masses[(masses >= mass_ini.min()) & (masses <= mass_ini.max())]

    # Interpolate sampled masses
    isoch_interp = []
    for isoch in isoch_binars:
        isoch_interp.append(interp1d(masses, mass_ini, isoch))
    isoch_interp = np.array(isoch_interp)

    return isoch_interp, masses


def interp1d(x_new, x, y):
    """
    Stripped down version of scipy.interpolate.interp1d. Assumes sorted
    'x' data.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)`` using some new values
    y_new = f(x_new)

    x_new.shape --> M
    x.shape     --> N
    y.shape     --> (D, N)
    y_new.T.shape --> (D, M)
    """

    _y = y.T

    # Find where in the original data, the values to interpolate
    # would be inserted.
    # Note: If x_new[n] == x[m], then m is returned by searchsorted.
    x_new_indices = np.searchsorted(x, x_new)

    # Clip x_new_indices so that they are within the range of
    # self.x indices and at least 1. Removes mis-interpolation
    # of x_new[n] = x[0]
    # x_new_indices = x_new_indices.clip(1, len(x) - 1).astype(int)

    # Calculate the slope of regions that each x_new value falls in.
    lo = x_new_indices - 1
    # hi = x_new_indices

    x_lo = x[lo]
    x_hi = x[x_new_indices]
    y_lo = _y[lo]
    y_hi = _y[x_new_indices]

    # Note that the following two expressions rely on the specifics of the
    # broadcasting semantics.
    slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

    # Calculate the actual value for each entry in x_new.
    y_new = slope * (x_new - x_lo)[:, None] + y_lo

    return y_new.T
