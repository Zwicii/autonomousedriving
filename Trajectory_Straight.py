import numpy as np
from numpy import arctan2


def referenceTrajectory(p, sp):
    # generation of the reference trajectory = straight Line

    n = np.int((sp.tf - sp.t0) / sp.dt)  # number of samples (here each interval one sample to enable simplified error)
    ds = p.vx * sp.dt  # approximate distance travelled in one time inerval assuming const. vel
    referenceTraj = np.zeros([n, 3])  # reserve space

    for i in np.arange(np.size(referenceTraj, 0)):
        # X
        referenceTraj[i, 0] = i * ds + p.ref_l
        # Y
        referenceTraj[i, 1] = 0
        # Orientation
        dydx = 0
        referenceTraj[i, 2] = arctan2(dydx, 1)

    return referenceTraj
