
from astropy.io import ascii
import numpy as np
from .HARDCODED import logAge_name, label_name, mag_name, col_name,\
    e_mag_name, e_col_name


def loadClust(clust_name, max_e=0.3):
    """
    Load observed cluster
    """
    cluster = ascii.read("input/{}.dat".format(clust_name))
    print("{} total stars".format(len(cluster)))
    try:
        msk1 = cluster[mag_name].mask
    except AttributeError:
        msk1 = np.array([])
    try:
        msk2 = cluster[col_name].mask
    except AttributeError:
        msk2 = np.array([])
    # msk2 = (cluster['e_Gmag'] >= max_e) | (cluster['e_BP-RP'] >= max_e)

    msk = np.array([])
    if msk1.any() and msk2.any():
        msk = msk1 | msk2
    elif msk1.any() and not msk2.any():
        msk = msk1
    elif not msk1.any() and msk2.any():
        msk = msk2

    if msk.any():
        print("{} masked stars".format(msk.sum()))
        cluster = cluster[~msk]
    cluster = np.array([
        cluster[mag_name], cluster[col_name], cluster[e_mag_name],
        cluster[e_col_name]])
    print("{} final members".format(cluster.shape[-1]))
    return cluster


def loadIsoch(idx_header, met, age, MS_end_ID=2):
    """
    HARDCODED to work with Gaia data

    MS_end_ID: This label marks the end of the MS, i.e. the turn-off point
    """
    # Load CMD isochrone (remove '#' from columns' names) IMPORTANT
    isoch = ascii.read("./isochs/{}.dat".format(
        str(met).replace(".", "_")), header_start=idx_header)

    msk = isoch[logAge_name] == age
    isoch = isoch[msk]

    # Remove TPAGB and post-AGB phases
    msk = (isoch['label'] == 8) | (isoch['label'] == 9)
    isoch = isoch[~msk]
    print("TPAGB and post-AGB phases removed from isochrone")

    # Identify the turn-off point
    idx = np.searchsorted(isoch[label_name], MS_end_ID)
    turn_off = isoch[mag_name][idx]

    return isoch, turn_off
