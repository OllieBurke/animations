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
sys.path.insert(0, "../python_fct")
from GWdata_analysis import *

sys.path.insert(0, "../kludge_fct/")
from gw_nk import *


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
coef_res0 = -0.00
coef_res1 = -0.00
coef_res2 = -0.00
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
def func(num, dataSet, line, axx):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, 0:num+delay])    

    return line
 

dataSet = np.array([time/3600, h_p_true, h_p_true])
numDataPoints = len(z)
 
# GET SOME MATPLOTLIB OBJECTS
fig, ax = plt.subplots(figsize=(16,6))
plt.ylabel(r'$h(t)$', fontdict=dict(fontsize=20) )
plt.xlabel(r'$t$ [hours]',fontdict=dict(fontsize=20) )
# ax = Axes3D(fig)
plt.axis('off')

 
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(dataSet[0], dataSet[1], lw=4, c='g')[0] # For line plot


# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line, ax), interval=5, blit=False)
# plt.show()

from matplotlib.animation import PillowWriter, FFMpegWriter


# line_ani.save('move.gif', writer=PillowWriter(fps=50))
writer = FFMpegWriter(fps=50)
line_ani.save('move.mp4',writer=writer)

