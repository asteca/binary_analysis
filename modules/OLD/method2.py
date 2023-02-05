
import numpy as np
# from scipy.optimize import differential_evolution as DE
# from . import binarFunc, uncertanties, likelihood
# from modules.HARDCODED import gamma_q_min, gamma_q_max, DE_tol


def InvPowerLaw(gamma_q, masses_mask, ran_x):
    """
    """
    # msum = masses_mask[0].sum()+masses_mask[1].sum()+masses_mask[2].sum()+masses_mask[3].sum()
    # if msum != len(ran_x):
    #     print("M2: inconsistent length")
    #     breakpoint()

    # This sampling does not respect the positions in 'masses_mask'. It just
    # selects groups of random numbers in the [0, 1] range and obtains their
    # (1/gamma_i) power. Each of these values is then added to the 'q_vals'
    # list with no regard for the 'masses_mask' order.
    #
    # This method depends on the number of bins used to generate the mass
    # intervals (used to obtain 'masses_mask')
    #
    # Combined with the *NOT mass-ordered* 'clust_synth', this is the method
    # that gives the best results so far (?!!)
    #
    q_vals = []
    for i, msk in enumerate(masses_mask):
        q_vals += list(ran_x[msk]**(1 / gamma_q[i]))
    q_vals = np.array(q_vals)

    # This simply assigns random numbers in the [0, 1] range and has the
    # worst performance of the three approaches.
    # q_vals = 1 * ran_x

    # if len(q_vals) != len(ran_x):
    #     print("M2: q_vals shape mismatch")
    #     breakpoint()

    return q_vals
