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

# for importing the external functions–
import sys


# Initial conditions
pstart = 12
eta_start = np.log(1e-5)
M_Kerr = np.log(1e6)
estart =  0.3
psi0 = 0
iotastart =0.2
chi0 = 0.0
a =0.9
# coefficients of: Resonantly enhanced and diminished strong-field gravitational-wave fluxes arXiv:1208.3906v2
# we could make this coefficients an order of magnitude bigger in order to correct for the duration of the resonance
tmax= 3e4
dt = 10
# Give me data ;)
switch_data_insp = 1

Var = np.array([pstart, eta_start, M_Kerr, estart, psi0, iotastart, chi0, a, coef_res0, coef_res1, coef_res2])

h_p_true, h_c_true, data, start_duration_res = EMRI_NK(Var, tmax, dt, 1)

time = data[:,0]; r = data[:,1]; theta = data[:,2]
omega_r = data[:,3]
omega_theta = data[:,4]
omega_phi = data[:,5]
E = data[:,6]
L_z = data[:,7]
Q = data[:,8]
psi = data[:,9]
chi = data[:,10]
phi = data[:,11]
p = data[:,12]
ecc = data[:,13]
iota = data[:,14]
dtau = data[:,15]
#
start_res = start_duration_res[0]
duration_res = start_duration_res[1]

# index of the resonance
i_res = np.where((time > start_res) & (time < (start_res + duration_res)))
print(np.shape(i_res))
print(omega_theta[i_res] /omega_r[i_res])

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

