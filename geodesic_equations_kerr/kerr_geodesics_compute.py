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
from scipy.optimize import root_scalar
from kerr_funs import roots_radial_potential, roots_z_equation, radial_potential
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, Y_to_xI, get_kerr_geo_constants_of_motion

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
dt = 10

# Build trajectory - AAK5PN waveform
traj_module = EMRIInspiral(func = "pn5",integrate_backwards = False)
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_module(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, max_init_len = int(1e4), DENSE_STEPPING = 1, in_coordinate_time=False)


p_interp = interp1d(t_traj, p_traj, kind = 'cubic') 
e_interp = interp1d(t_traj, e_traj, kind = 'cubic') 

breakpoint()
x_traj = Y_to_xI(a, p_traj, e_traj, Y_traj)
E,L, Q = get_kerr_geo_constants_of_motion(a, p_traj, e_traj, x_traj)

E_interp = interp1d(t_traj, E, kind = 'cubic') 
L_interp = interp1d(t_traj, L, kind = 'cubic') 
Q_interp = interp1d(t_traj, Q, kind = 'cubic') 
# =========== Stick to geodesic for now! =================

# initial geodesic values 
E0 = E[0]
L0 = L[0]
Q0 = Q[0]

# Radial potential? 


# Specify the values for a, E, L, and Q
E0 = E[0]
L0 = L[0]
Q0 = Q[0]

# Find the roots using the 
r_a, r_p, r_3, r_4 = roots_radial_potential(a,E0,L0,Q0)

z_p, z_m = roots_z_equation(a, E0,L0,Q0)
def z_chi(z_m,chi):
    return z_m * np.cos(chi)**2
def deriv_psi_t(psi,chi, p,e,a, E, L, Q):
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    gam = E*(((r**2 + a**2)**2)/delta - a**2) - 2*r*a*L/delta # gamma value - correct

    _, _, r_3, r_4 = roots_radial_potential(a,E,L, Q)   # Define roots of radial potential
    _, z_m = roots_z_equation(a, E, L, Q)               # Define roots of 

    first_term_num = np.sqrt(1 - E**2)* ((p - r_3*(1-e)) - e * (p + r_3*(1 - e)*np.cos(psi)))**(1/2) # Looks OK
    second_term_num =  np.sqrt( (p - r_4*(1+e)) + e*(p - r_4*(1 + e)*np.cos(psi))) # Looks OK 
    denom = (1 - e**2) * (gam + a**2 * E * z_chi(z_m,chi))  # 

    return first_term_num * second_term_num / denom
def deriv_chi_t(psi,chi, p,e,a, E, L, Q):
    """
    checked - derivative of chi with respect to t - determimines 
    """
    z_p, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    r = p/(1 + e*np.cos(psi)) # Radial coordinate
     
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr
    gam = E*(((r**2 + a**2)**2)/delta - a**2) - 2*r*a*L/delta # gamma value
    beta = a**2 * (1 - E**2)

    numerator = np.sqrt(beta * (z_p - z_chi(z_m,chi)))
    denom = gam + a**2 * E * z_chi(z_m, chi)
    return numerator/denom

def deriv_phi_t(psi,chi, p,e,a, E, L, Q):
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    z_p, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    
    first_term = (2 * r * (E * (r**2 + a**2) - L * a)) 
    second_term = - (a**2 * (E**2 - delta) + (L - a * E)**2) * (1 - z_chi(z_m,chi)**2) 
    denom = (delta * (1 - z_chi(z_m,chi)**2)) 
    return (first_term + second_term)/denom

def Kerr_geodesic_eqn(t, y, p , e , a, E, L, Q):
    psi, chi, phi = y

    dpsi_dt = deriv_psi_t(psi,chi, p,e,a, E, L, Q)
    dchi_dt = deriv_chi_t(psi,chi, p,e,a, E, L, Q) 
    dphi_dt = deriv_phi_t(psi,chi, p,e,a, E, L, Q)

    return [dpsi_dt, dchi_dt, dphi_dt]

t_start = 0
t_end = 10000
t_span = (t_start,t_end)
t_eval = np.arange(t_start, t_end,0.001)

psi0 = 0 # Here we start at periastron

theta0 = np.pi/3 # Here we start at theta = np.pi/3

z_p, z_m = roots_z_equation(a, E0, L0, Q0)
chi0 = (np.cos(theta0)**2 / z_m)**(1/2) # Starting value for chi0

phi0 = 0

y0 = [psi0, chi0,phi0]
solution = solve_ivp(Kerr_geodesic_eqn, t_span, y0, args=(p0, e0, a, E0, L0, Q0),t_eval = t_eval)

t = solution.t
psi_sol = solution.y[0]
chi_sol = solution.y[1]
phi_sol = solution.y[2]

r = p0/(1 + e0*np.cos(psi_sol))
theta_sol = z_m * np.cos(chi_sol)**2

plt.plot(t,r);plt.show();plt.clf()
plt.plot(t,theta_sol);plt.show()
breakpoint()