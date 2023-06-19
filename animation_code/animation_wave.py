

# Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#https://stackoverflow.com/questions/48152754/matplotlib-plot-points-over-time-where-old-points-fade
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import subprocess
import os
import sys
sys.path.append("../geodesic_equations_kerr/")

# for importing the external functions–
#plt.rcParams['axes.facecolor'] = 'none'

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from kerr_funcs import roots_z_equation, deriv_psi_t, deriv_chi_t, deriv_phi_t
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import Y_to_xI, get_kerr_geo_constants_of_motion, get_separatrix

# for importing the external functions–
import sys


# Initial conditions
M = 1e6
mu = 10.0
a = 0.9
p0 = 12.0
e0 =  0.001
iota0 = 0.5
Y0 = np.cos(iota0)

Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0 

# Set up length of trajectory in time (seconds)
T = 1e5 / (365*24*60*60)
dt = 10 

# Build trajectory - AAK5PN waveform - Using time in [M}]
traj_module = EMRIInspiral(func = "pn5")
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_module(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, max_init_len = int(1e7), DENSE_STEPPING = 1, in_coordinate_time=False)

dist = 1
qS = 0.5
phiS = 0.5
qK = 0.5
phiK = 0.5

# Set up inspiral_kwargs - note use of "integrate_backwards"
inspiral_kwargs = {
    "max_init_len": int(1e8),
    "DENSE_STEPPING": 0,
    "err": 1e-10, 
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e5),  # all of the trajectories will be well under len = 1000
    "use_gpu": False  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": False,  # GPU is availabel for this type of summation
    "pad_output": False
    }

wave_generator = GenerateEMRIWaveform('Pn5AAKWaveform', inspiral_kwargs = inspiral_kwargs, amplitude_kwargs = amplitude_kwargs,
                                                        Ylm_kwargs = Ylm_kwargs, sum_kwargs = sum_kwargs)
waveform_EMRI = wave_generator(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, 
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T) 

h_p_true = waveform_EMRI.real
time = np.arange(0,dt*len(h_p_true),dt)


delay=0
# ANIMATION FUNCTION
def func(num, dataSet, line, axx):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, 0:num+delay])    

    return line
 

dataSet = np.array([time/360, h_p_true, h_p_true])
numDataPoints = len(time)
 
# GET SOME MATPLOTLIB OBJECTS
fig, ax = plt.subplots(figsize=(16,6))
plt.ylabel(r'$h_{+}(t)$', fontdict=dict(fontsize=20) )
plt.xlabel(r'$t$ [hours]',fontdict=dict(fontsize=20) )
# ax = Axes3D(fig)
# plt.axis('off')


 
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(dataSet[0], dataSet[1], lw=4, c='g')[0] # For line plot


# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line, ax), interval=5, blit=False)
# plt.show()

from matplotlib.animation import PillowWriter, FFMpegWriter

# Save the animation as a video file
writer = animation.PillowWriter(fps=50, metadata=dict(artist='OllieBurke'))
print("Now running!")
line_ani.save('Kerr_Traj_p0_12_e0_0_iota0_0p3.gif', writer=writer)
result = subprocess.run("mv *.gif waveform_gif", shell=True, capture_output=True, text=True)