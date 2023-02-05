
import numpy as np
from . import extin_coefs
from . import readData


def isochProcess(cmd_systs, idx_header, met, age, ext, dist):
    """
    Load the corresponding isochrone, move it, interpolate it, and return
    the required arrays.
    """
    # turn_off : approximated magnitude value of the turn off point for the
    # isochrone
    isoch, turn_off = readData.loadIsoch(idx_header, met, age)

    # Move isochrone.
    turn_off += dist
    isoch_mv = move(cmd_systs, isoch, ext, dist)

    # Interpolate extra points
    isoch_interp = interp(np.concatenate([isoch_mv, [isoch['Mini']]]))

    # (magnitude and color), (magnitudes for the colors), (initial mass)
    isoch_phot = np.concatenate([[isoch_interp[0]], [isoch_interp[-2]]])
    mass_ini = isoch_interp[-1]

    # Used for binary masses estimation
    isoch_col_mags = isoch_interp[1:3]

    return turn_off, isoch_phot, mass_ini, isoch_col_mags


def move(cmd_systs, isoch, ext, dist, R_V=3.1):
    """
    """
    # Extinction coefficients defined for the Gaia EDR3 system
    ext_coefs = extin_coefs.main(cmd_systs)

    isochrone = (
        isoch['Gmag'], isoch['G_BPmag'], isoch['G_RPmag'],
        isoch['G_BPmag'] - isoch['G_RPmag'])

    Av = R_V * ext
    N_fc = (3, 1)  # HARDCODED
    Nf, Nc = N_fc

    def magmove(fi, mag):
        Ax = (ext_coefs[fi][0] + ext_coefs[fi][1] / R_V) * Av
        return np.array(mag) + dist + Ax

    def colmove(ci, col):
        Ex = (
            (ext_coefs[Nf + ci][0][0] + ext_coefs[Nf + ci][0][1] / R_V)
            - (ext_coefs[Nf + ci][1][0] + ext_coefs[Nf + ci][1][1] / R_V)) * Av
        return np.array(col) + Ex

    iso_moved = []
    # Move filters.
    for fi, mag in enumerate(isochrone[:Nf]):
        iso_moved.append(magmove(fi, mag))
    # Move colors.
    for ci, col in enumerate(isochrone[Nf:(Nf + Nc)]):
        iso_moved.append(colmove(ci, col))

    return np.array(iso_moved)


def interp(isoch, N=5000):
    interp_data = []
    for fce in isoch:
        t, xp = np.linspace(0., 1., N), np.linspace(0, 1, len(fce))
        interp_data.append(np.interp(t, xp, fce))
    return (np.array(interp_data))
