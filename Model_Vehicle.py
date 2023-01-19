# -*- coding: utf-8 -*-
from enum import Enum

import numpy as np
from numpy import cos, sin, tan

import matplotlib.pyplot as plt
import matplotlib.animation
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std

#plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
plt.rcParams["animation.html"] = "jshtml"

def ode(x, t, p):
    """Function of the robots kinematics
    Args:
        x: state  X, Y, Theta
        t: time
        p(object): parameter container class
    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = p.u  # control(x, t)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / p.l * u1 * tan(u2)])

    return dxdt  # return state derivative


class Model(Enum):
    KS = 1
    ST = 2
    STD = 3
    MB = 4


def odeCustom(x, t, p, model):
    """Function of the robots kinematics
    Args:
        x: state  X, Y, Theta
        t: time
        p(object): parameter container class
    Returns:
        dxdt: state derivative
    """
    #x1, x2, x3 = x  # state vector
    #vel = p.vel  # control(x, t)  # control vector
    #steering_angle = p.steering
    # dxdt = f(x, u):
    #dxdt = np.array([u1 * cos(x3),
    #                 u1 * sin(x3),
    #                 1 / p.l * u1 * tan(u2)])
    x[2] = p.steering_angle
    x[3] = p.vel

    # x: kinmatic parameters
    # x[2] = steering angle
    # x[3] = velocity
    # uInit[0] = steering velocity
    # uInit[1] = acceleration

    # vehicle_dynamics_mb(x, uInit, p):


    ACC = -5;


    if model == Model.KS :
        return vehicle_dynamics_ks(x, [p.turn_rate, ACC], p.p)  # return state derivative
    elif model == Model.MB :
        return vehicle_dynamics_mb(x, [p.turn_rate, ACC], p.p)  # return state derivative
    elif model == Model.STD:
        return vehicle_dynamics_std(x, [p.turn_rate, ACC], p.p)  # return state derivative
    elif model == Model.ST:
        return vehicle_dynamics_st(x, [p.turn_rate, ACC], p.p)  # return state derivative
    else:
        None # return state derivative


def plot_data(x, u, r, t, fig_width, fig_height, ofileName, save=False):
    """Plotting function of simulated state and actions

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        r(ndarray): reference
        t(ndarray): time vector
        fig_width: figure width in cm
        fig_height: figure height in cm
        save (bool) : save figure (default: False)
    Returns: None

    """
    # creating a figure with 3 subplots, that share the x-axis
    fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # set figure size to desired values
    fig1.set_size_inches(fig_width / 2.54, fig_height / 2.54)

    # plot y_1 in subplot 1
    ax1.plot(t, x[:, 0], label='$x(t)$', lw=1, color='r')

    # plot y_2 in subplot 1
    ax1.plot(t, x[:, 1], label='$y(t)$', lw=1, color='b')

    # plot theta in subplot 2
    ax2.plot(t, np.rad2deg(x[:, 2]), label=r'$\theta(t)$', lw=1, color='g')

    # plot control in subplot 3, left axis red, right blue
    ax3.plot(t, u[:, 1], label=r'$v(t)$', lw=1, color='r')
    ax3.tick_params(axis='y', colors='r')
    ax33 = ax3.twinx()
    ax33.plot(t, np.rad2deg(u[:, 0]), label=r'$\phi(t)$', lw=1, color='b')
    ax33.spines["left"].set_color('r')
    ax33.spines["right"].set_color('b')
    ax33.tick_params(axis='y', colors='b')

    # Grids
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    # set the labels on the x and y axis and the titles
    ax1.set_title('Position coordinates')
    ax1.set_ylabel(r'm')
    ax1.set_xlabel(r't in s')
    ax2.set_title('Orientation')
    ax2.set_ylabel(r'deg')
    ax2.set_xlabel(r't in s')
    ax3.set_title('Velocity / steering angle')
    ax3.set_ylabel(r'm/s')
    ax33.set_ylabel(r'deg')
    ax33.set_xlabel(r't in s')

    # put a legend in the plot
    ax1.legend()
    ax2.legend()
    ax3.legend()
    li3, lab3 = ax3.get_legend_handles_labels()
    li33, lab33 = ax33.get_legend_handles_labels()
    ax3.legend(li3 + li33, lab3 + lab33, loc=0)

    # automatically adjusts subplot to fit in figure window
    plt.tight_layout()

    # save the figure in the working directory
    if save:
        plt.savefig(ofileName+'.pdf')  # save output as pdf
        plt.savefig(ofileName+'.pgf')  # for easy export to LaTex
    return None

def car_animation(x, u, r, t, p, x_traj, ofileName):
    """Animation function of the car-like mobile robot

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        r(ndarray): reference
        t(ndarray): time vector
        p(object): parameters

    Returns: None

    """
    # Setup two empty axes with enough space around the trajectory so the car
    # can always be completely plotted. One plot holds the sketch of the car,
    # the other the curve
    dx = 1.5 * p.l
    dy = 1.5 * p.l
    fig2, ax = plt.subplots()
    ax.set_xlim([min(min(min(x_traj[:, 0] - dx), -dx), min(r[:, 0])),
                 max(max(max(x_traj[:, 0] + dx), dx), max(r[:, 0]))])
    ax.set_ylim([min(min(min(x_traj[:, 1] - dy), -dy), min(r[:, 1])),
                 max(max(max(x_traj[:, 1] + dy), dy), max(r[:, 1]))])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    # Axis handles
    h_x_traj_plot, = ax.plot([], [], 'b')  # state trajectory in the y1-y2-plane
    h_xf_traj_plot, = ax.plot([], [], 'r')  # state trajectory in the y1-y2-plane
    h_r_traj_plot, = ax.plot([], [], 'g')
    h_car, = ax.plot([], [], 'k', lw=2)  # car

    def car_plot(x, u):
        """Mapping from state x and action u to the position of the car elements

        Args:
            x: state vector
            u: action vector

        Returns:

        """
        wheel_length = 0.1 * p.l
        y1, y2, theta = x
        v, phi = u

        # define chassis lines
        chassis_y1 = [y1, y1 + p.l * cos(theta)]
        chassis_y2 = [y2, y2 + p.l * sin(theta)]

        # define lines for the front and rear axle
        rear_ax_y1 = [y1 + p.w * sin(theta), y1 - p.w * sin(theta)]
        rear_ax_y2 = [y2 - p.w * cos(theta), y2 + p.w * cos(theta)]
        front_ax_y1 = [chassis_y1[1] + p.w * sin(theta + phi),
                       chassis_y1[1] - p.w * sin(theta + phi)]
        front_ax_y2 = [chassis_y2[1] - p.w * cos(theta + phi),
                       chassis_y2[1] + p.w * cos(theta + phi)]

        # define wheel lines
        rear_l_wl_y1 = [rear_ax_y1[1] + wheel_length * cos(theta),
                        rear_ax_y1[1] - wheel_length * cos(theta)]
        rear_l_wl_y2 = [rear_ax_y2[1] + wheel_length * sin(theta),
                        rear_ax_y2[1] - wheel_length * sin(theta)]
        rear_r_wl_y1 = [rear_ax_y1[0] + wheel_length * cos(theta),
                        rear_ax_y1[0] - wheel_length * cos(theta)]
        rear_r_wl_y2 = [rear_ax_y2[0] + wheel_length * sin(theta),
                        rear_ax_y2[0] - wheel_length * sin(theta)]
        front_l_wl_y1 = [front_ax_y1[1] + wheel_length * cos(theta + phi),
                         front_ax_y1[1] - wheel_length * cos(theta + phi)]
        front_l_wl_y2 = [front_ax_y2[1] + wheel_length * sin(theta + phi),
                         front_ax_y2[1] - wheel_length * sin(theta + phi)]
        front_r_wl_y1 = [front_ax_y1[0] + wheel_length * cos(theta + phi),
                         front_ax_y1[0] - wheel_length * cos(theta + phi)]
        front_r_wl_y2 = [front_ax_y2[0] + wheel_length * sin(theta + phi),
                         front_ax_y2[0] - wheel_length * sin(theta + phi)]

        # empty value (to disconnect points, define where no line should be plotted)
        empty = [np.nan, np.nan]

        # concatenate set of coordinates
        data_y1 = [rear_ax_y1, empty, front_ax_y1, empty, chassis_y1,
                   empty, rear_l_wl_y1, empty, rear_r_wl_y1,
                   empty, front_l_wl_y1, empty, front_r_wl_y1]
        data_y2 = [rear_ax_y2, empty, front_ax_y2, empty, chassis_y2,
                   empty, rear_l_wl_y2, empty, rear_r_wl_y2,
                   empty, front_l_wl_y2, empty, front_r_wl_y2]

        # set data
        h_car.set_data(data_y1, data_y2)

    def init():
        """Initialize plot objects that change during animation.
           Only required for blitting to give a clean slate.

        Returns:

        """
        h_x_traj_plot.set_data([], [])
        h_xf_traj_plot.set_data([], [])
        h_r_traj_plot.set_data([], [])
        h_car.set_data([], [])
        return h_x_traj_plot, h_car, h_xf_traj_plot, h_r_traj_plot

    def animate(i):
        """Defines what should be animated
        Args:
            i: frame number
        Returns:
        """
        k = i % len(t)
        ax.set_title('Time (s): ' + str(t[k]), loc='left')
        h_x_traj_plot.set_xdata(x[0:k, 0])
        h_x_traj_plot.set_ydata(x[0:k, 1])
        h_xf_traj_plot.set_xdata(x[0:k, 0] + p.l * cos(x[0:k, 2]))
        h_xf_traj_plot.set_ydata(x[0:k, 1] + p.l * sin(x[0:k, 2]))
        h_r_traj_plot.set_xdata(r[:, 0])
        h_r_traj_plot.set_ydata(r[:, 1])
        car_plot(x[k, :], u[k, :])
        return h_x_traj_plot, h_car, h_xf_traj_plot, h_r_traj_plot

    ani = matplotlib.animation.FuncAnimation(fig2, animate, init_func=init, frames=len(t) + 1,
                             interval=(t[1] - t[0]) * 1000,
                             blit=True)
    #ani.save(ofileName + '.mp4', writer='ffmpeg', fps=1 / (t[1] - t[0]))
    return ani
