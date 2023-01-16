
import numpy
from matplotlib.pyplot import title, legend
from scipy.integrate import odeint
import matplotlib as plt

from test_vehicle import func_MB, x0_MB, x0_ST, func_ST, func_KS, x0_STD, func_STD, x0_KS, p


def cornering_left(v_delta, a_long, tFinal):
    # steering to left
    t = numpy.arange(0, tFinal, 0.01)
    u = [v_delta, a_long]

    # simulate multibody
    x_left = odeint(func_MB, x0_MB, t, args=(u, p))

    # simulate single-track model
    x_left_st = odeint(func_ST, x0_ST, t, args=(u, p))

    # simulate kinematic single-track model
    x_left_ks = odeint(func_KS, x0_KS, t, args=(u, p))

    # simulate single-track drift model
    x_left_std = odeint(func_STD, x0_STD, t, args=(u,p))

    # results
    # position
    title('positions turning')
    plt.plot([tmp[0] for tmp in x_left], [tmp[1] for tmp in x_left])
    plt.plot([tmp[0] for tmp in x_left_st], [tmp[1] for tmp in x_left_st])
    plt.plot([tmp[0] for tmp in x_left_ks], [tmp[1] for tmp in x_left_ks])
    plt.plot([tmp[0] for tmp in x_left_std], [tmp[1] for tmp in x_left_std])
    legend(['MB', 'ST', 'KS', 'STD'])
    plt.autoscale()
    plt.show()

    # slip angle
    title('slip angle turning')
    plt.plot(t, [tmp[10] / tmp[3] for tmp in x_left])
    plt.plot(t, [tmp[6] for tmp in x_left_st])
    plt.plot(t, [tmp[6] for tmp in x_left_std])
    legend(['MB', 'ST', 'STD'])
    plt.show()

    return x_left
