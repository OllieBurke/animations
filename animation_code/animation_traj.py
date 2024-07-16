

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
from few.utils.utility import Y_to_xI, get_kerr_geo_constants_of_motion, get_separatrix

# for importing the external functions–
import sys


# Initial conditions
M = 1e6
mu = 10.0
a = 0.9
p0 = 12.0
e0 =  0.3
iota0 = 0.3
Y0 = np.cos(iota0)

Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0 

# Set up length of trajectory in time (seconds)
T = 1e4 / (365*24*60*60)
dt = 50 

# Build trajectory - AAK5PN waveform - Using time in [M}]
traj_module = EMRIInspiral(func = "pn5")
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_module(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, max_init_len = int(1e7), DENSE_STEPPING = 1, in_coordinate_time=False)

# Build interpolants for trajectories
p_interp = interp1d(t_traj, p_traj, kind = 'cubic') 
e_interp = interp1d(t_traj, e_traj, kind = 'cubic') 

x_traj = Y_to_xI(a, p_traj, e_traj, Y_traj)
print("Separatrix located at ",get_separatrix(a, e_traj[-1], x_traj[-1]))
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

# Due to numerical error in FEW, we force Q = 0 for purely equatorial orbits
if (iota0) < 1e-2: # Equatorial - no inclination
    chi0 = np.pi/2 # Starting value on equatorial plane
    Q = np.zeros(len(Q))
    Q_interp = interp1d(t_traj,Q, kind = 'linear')
else:
    chi0 = iota0
phi0 = 0 # Somewhat arbitrary


# Store initial conditions
y0 = [psi0, chi0,phi0] 
# Integrate geodesic equations
solution = solve_ivp(Kerr_geodesic_eqn, t_span, y0, args=(p_interp, e_interp, a, E_interp, L_interp, Q_interp), t_eval = t_eval)

# Extract solutions
t_M = solution.t
psi_sol = solution.y[0]
chi_sol = solution.y[1]
phi_sol = solution.y[2]

# Build evolution of integrals of motion
E_sol = E_interp(t_M)
L_sol = L_interp(t_M)
Q_sol = Q_interp(t_M)

# Convert to BL coordinates r and theta
# radial coordinate r
r_sol = p_interp(t_M)/(1 + e_interp(t_M)*np.cos(psi_sol))

# polar angle theta
z_m_traj = []
for E_val, L_val, Q_val in zip(E_sol,L_sol,Q_sol):
    _, z_m = roots_z_equation(a, E_val, L_val, Q_val)
    z_m_traj.append(z_m)

z_m_traj = np.array(z_m_traj)

theta_sol = np.arccos(np.sqrt(z_m_traj) * np.cos(chi_sol))

# Change to flat space, spherical polar coordinates
x = r_sol*np.sin(theta_sol)*np.cos(phi_sol)
y = r_sol*np.sin(theta_sol)*np.sin(phi_sol)

if iota0 < 1e-2: # For equatorial orbits. Need this for plotting!
    z = 0.25 + r_sol*np.cos(theta_sol)
else:
    z = r_sol*np.cos(theta_sol)

Plot_Orbit = False
if Plot_Orbit:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.scatter([0], [0], [0.25], color='black', s = 100, marker='o',)  
    ax.set_xlabel(r'$r\sin\theta\cos\phi$')
    ax.set_ylabel(r'$r\sin\theta\sin\phi$')
    ax.set_zlabel(r'$r\cos\theta$')
    plt.savefig("Radiation_Reaction_Orbit.png")
    plt.show()

# delay = 5 # Length of the compact objects trail

# Animation function
def func(num, dataSet, line, points, axx):
    # Need these two lines if we want to see the full trajectory!
    delay = num 
    num = 0
    line.set_data(dataSet[0:2, num:num+delay])          # Actually plots the data 
    line.set_3d_properties(dataSet[2, num:num+delay])

    # points.set_data(dataSet[0:2, num+delay-1:num+delay])
    points.set_3d_properties(dataSet[2, num+delay-1:num+delay])

    # Check the condition x > 0 and y > 0
    try:
        if dataSet[0, num+delay-1] > 0 and dataSet[1, num+delay-1] > 0:
            # If in front of black hole
            line.set_zorder(0)
            points.set_zorder(0)
        else: #if behind black hole
            line.set_zorder(2)
            points.set_zorder(2)
    except IndexError:
        pass
    axx.view_init(elev=16., azim=num/10)  # azim -> how slowly the plot rotates
    
    return line
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

dataSet = np.array([x, y, z])
numDataPoints = len(z)

# Get some Matplotlib objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create Line3D objects instead of Line2D objects
line = Line3D([], [], [], color='red', linewidth=2)
point = Line3D([], [], [], marker='o', color='blue')

# Add the line to the plot
ax.add_line(line)
# Add Line3D objects to the plot
if iota0 < 1e-2:
    ax.plot([0],[0],[0.25],'ko',ms=10, zorder = 0)
else:
    ax.plot([0],[0],[0],'ko',ms=10, zorder = 0)

ax.add_line(line)
ax.add_line(point)

# Axes properties
ax.set_xlabel(r'$r\sin\theta\cos\phi$')
ax.set_ylabel(r'$r\sin\theta\sin\phi$')
ax.set_zlabel(r'$r\cos\theta$')

ax.set_xlim([min(x),max(x)])
ax.set_ylim([min(y),max(y)])
if iota0 > 1e-2:
    ax.set_zlim([min(z),max(z)])


ax.set_title('Weak Field: Eccentric/Inclined orbit into a rotating black hole\n$M = 10^{6}M_{\odot}$, $\mu = 10M_{\odot}$, $a = 0.9$, $p_{0} = 12.0$, $e_{0} = 0.3$, $\iota_{0} = 0.3$')
plt.tight_layout()
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, line, point, ax), interval=5, blit=False)

# Save the animation as a video file
writer = animation.PillowWriter(fps=50, metadata=dict(artist='Your Name'))
print("Now running!")
line_ani.save('Kerr_Traj_full_weak_field_large_dt.gif', writer=writer)
result = subprocess.run("mv *.gif trajectory_gif", shell=True, capture_output=True, text=True)
