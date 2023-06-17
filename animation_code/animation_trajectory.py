import matplotlib as mpl 
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['axes.facecolor'] = 'none'

import matplotlib.animation as animation
#https://stackoverflow.com/questions/48152754/matplotlib-plot-points-over-time-where-old-points-fade
import glob
import numpy as np
from numpy.linalg import inv
from scipy import signal
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
import imageio

import os
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, Y_to_xI

# for importing the external functions–
import sys


# Initial conditions
M = 1e6
mu = 10.0
a = 0.9
p0 = 12.0
e0 =  0.3
iota0 = 0.5
Y0 = np.cos(iota0)

Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0 

# Set up length of trajectory in time

T = 3e4 / (365*24*60*60)
breakpoint()
dt = 10

# Build trajectory - AAK5PN waveform
traj_module = EMRIInspiral(func = "pn5",integrate_backwards = False)
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_module(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, max_init_len = int(1e4), DENSE_STEPPING = 1)
breakpoint()

# time = data[:,0]; r = data[:,1]; theta = data[:,2]
# omega_r = data[:,3]
# omega_theta = data[:,4]
# omega_phi = data[:,5]
# E = data[:,6]
# L_z = data[:,7]
# Q = data[:,8]
# psi = data[:,9]
# chi = data[:,10]
# phi = data[:,11]
# p = data[:,12]
# ecc = data[:,13]
# iota = data[:,14]
# dtau = data[:,15]

time = tvec
theta = Phi_theta_traj
phi = Phi_phi_traj
p = p_traj
ecc = e_traj


x=r*np.sin(theta)*np.cos(phi)
y= r*np.sin(theta)*np.sin(phi)
z=r*np.cos(theta)


delay=200
# ANIMATION FUNCTION
def func(num, dataSet, line, points, axx):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, num:num+delay])    
    line.set_3d_properties(dataSet[2, num:num+delay])    

    points.set_data(dataSet[0:2, num+delay-1:num+delay])
    points.set_3d_properties(dataSet[2, num+delay-1:num+delay])

    axx.view_init(elev=16., azim=num/10)
    return line
 

dataSet = np.array([x, y, z])
numDataPoints = len(z)
 
# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure()
ax = Axes3D(fig)
 
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0] # For line plot

point = plt.plot(dataSet[0], dataSet[1], dataSet[2], 'ko')[0]

# AXES PROPERTIES]


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


a = Arrow3D([0.0, 0.0], [0.0, 0.0], 
                [1.0, 2.0], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="r")

ax.add_artist(a)
ax.plot([0],[0],[0],'ko',ms=50)
ax.view_init(18, 160)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Trajectory of electron for E vector along [120]')

# white background
# ax.axis('off')

# make the panes transparent
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# # make the grid lines transparent
# ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line, point, ax), interval=5, blit=False)
# plt.show()

from matplotlib.animation import PillowWriter, FFMpegWriter


# line_ani.save('move.gif', writer=PillowWriter(fps=50))
writer = FFMpegWriter(fps=50)
line_ani.save('move.mp4',writer=writer)

