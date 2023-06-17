import matplotlib as mpl 
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['axes.facecolor'] = 'none'

import matplotlib.animation as animation
#https://stackoverflow.com/questions/48152754/matplotlib-plot-points-over-time-where-old-points-fade
import numpy as np

from kerr_funs import roots_z_equation, deriv_psi_t, deriv_chi_t, deriv_phi_t
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import Y_to_xI, get_kerr_geo_constants_of_motion

# for importing the external functions–
import sys


# Initial conditions
M = 1e6
mu = 10.0
a = 0.9
p0 = 5.0
e0 =  0.5
iota0 = 0.5
Y0 = np.cos(iota0)

Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0 

# Set up length of trajectory in time (seconds)
T = 1e4 / (365*24*60*60)
dt = 1 

# Build trajectory - AAK5PN waveform - Using time in [M}]
traj_module = EMRIInspiral(func = "pn5",integrate_backwards = False)
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_module(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, max_init_len = int(1e7), DENSE_STEPPING = 1, in_coordinate_time=False)

# Build interpolants for trajectories
p_interp = interp1d(t_traj, p_traj, kind = 'cubic') 
e_interp = interp1d(t_traj, e_traj, kind = 'cubic') 

x_traj = Y_to_xI(a, p_traj, e_traj, Y_traj)
E,L, Q = get_kerr_geo_constants_of_motion(a, p_traj, e_traj, x_traj)

# Build interpolants for "constants" of motion
E_interp = interp1d(t_traj, E, kind = 'cubic') 
L_interp = interp1d(t_traj, L, kind = 'cubic') 
Q_interp = interp1d(t_traj, Q, kind = 'cubic') 

# initial geodesic values 
E0 = E_interp(0)
L0 = L_interp(0)
Q0 = Q_interp(0)


def Kerr_geodesic_eqn(t, y, p , e , a, E, L, Q):
    """
        Computes the derivatives of the Kerr geodesic equations at a given time.

        The function calculates the derivatives of the Kerr geodesic equations for the given time `t` and the current
        values of the variables `y` (psi, chi, phi). The function also takes as input various parameters p, e, a, E, L, and Q,
        which may be functions of time.

        Parameters:
            t (float): The current time.
            y (list): The current values of the variables [psi, chi, phi].
            p (function): A function that returns the value of p as a function of time.
            e (function): A function that returns the value of e as a function of time.
            a (float): The value of the Kerr parameter.
            E (function): A function that returns the value of E as a function of time.
            L (function): A function that returns the value of L as a function of time.
            Q (function): A function that returns the value of Q as a function of time.

        Returns:
            list: The derivatives of the variables [dpsi_dt, dchi_dt, dphi_dt] at the given time.

        """
    # Take in interpolants
    p = p(t)
    e = e(t)
    E = E(t)
    L = L(t)
    Q = Q(t)

    # Set initial conditions 
    psi, chi, phi = y

    # Define derivatives
    dpsi_dt = deriv_psi_t(t,psi,chi, p,e,a, E, L, Q)
    dchi_dt = deriv_chi_t(t,psi,chi, p,e,a, E, L, Q) 
    dphi_dt = deriv_phi_t(t,psi,chi, p,e,a, E, L, Q)

    return [dpsi_dt, dchi_dt, dphi_dt]

# MUST BE IN COORDINATE TIME - define start and end of trajectory
t_start = 0
t_end = 1*t_traj[-1]  
dt_M = dt/(M*4.92e-6)      # Define spacing in units of M

t_span = (t_start,t_end)
t_eval = np.arange(t_start, t_end,dt_M)

psi0 = 0 # Here we start at periastron
theta0 = iota0 # Here we start at theta = iota0. Not sure this is correct

z_p, z_m = roots_z_equation(a, E0, L0, Q0)
chi0 = (np.cos(theta0)**2 / z_m)**(1/2) # Starting value for chi0

phi0 = 0 # Somewhat arbitrary

# Store initial conditions
y0 = [psi0, chi0,phi0] 
# Integrate geodesic equations
solution = solve_ivp(Kerr_geodesic_eqn, t_span, y0, args=(p_interp, e_interp, a, E_interp, L_interp, Q_interp),t_eval = t_eval)

# Extract solutions
t_M = solution.t
psi_sol = solution.y[0]
chi_sol = solution.y[1]
phi_sol = solution.y[2]

# Convert to BL coordinates r and theta
r_sol = p_interp(t_M)/(1 + e_interp(t_M)*np.cos(psi_sol))
theta_sol = np.arccos(np.sqrt(z_m) * np.cos(chi_sol))

# Change to flat space, spherical polar coordinates
x_sol = r_sol*np.sin(theta_sol)*np.cos(phi_sol)
y_sol = r_sol*np.sin(theta_sol)*np.sin(phi_sol)
z_sol = r_sol*np.cos(theta_sol)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sol, y_sol, z_sol)
ax.scatter([0], [0], [0], color='black', s = 100, marker='o',)  # Add a red dot at x=y=z=0
ax.set_xlabel(r'$r\sin\theta\cos\phi$')
ax.set_ylabel(r'$r\sin\theta\sin\phi$')
ax.set_zlabel(r'$r\cos\theta$')
plt.show()
