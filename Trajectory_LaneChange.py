import numpy as np
from numpy import arctan2


def referenceTrajectory(p, sp):
    # generation of the reference trajectory

    n = np.int((sp.tf - sp.t0) / sp.dt)  # number of samples (here each interval one sample to enable simplified error)
    ds = p.vx * sp.dt  # approximate distance travelled in one time inerval assuming const. vel
    referenceTraj = np.zeros([n, 3])  # reserve space

    k1 = 0.3  # scaling parameter for lane-change (scale tanh input)
    k2 = 5  # offset parameter for lane-change (offset tanh input)
    manouverWidth = 3  # maneuver width in [m]
    for i in np.arange(np.size(referenceTraj, 0)):
        # X
        referenceTraj[i, 0] = i * ds + p.ref_l
        # Y
        referenceTraj[i, 1] = manouverWidth * 0.5 * (np.tanh(ds * i * k1 - k2) + 1)
        # Orientation
        dydx = manouverWidth * 0.5 * k1 / ((np.cosh(k2 - ds * i * k1)) * (np.cosh(k2 - ds * i * k1)))
        referenceTraj[i, 2] = arctan2(dydx, 1)

    return referenceTraj
