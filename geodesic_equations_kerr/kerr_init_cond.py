from kerr_funcs import roots_z_equation, deriv_psi_t, deriv_chi_t, deriv_phi_t

import numpy as np

from scipy import integrate

# Define the functions to compute the integrals
def I_psi(t_start, psi0, chi0, p0, e0, a, E0, L0, Q0):
    result, _ = integrate.quad(lambda psi: 1/deriv_psi_t(t_start, psi, chi0, p0, e0, a,  E0, L0, Q0),0,psi0)
    return result

def I_chi(t_start, psi0, chi0, p0, e0, a, E0, L0, Q0):
    result, _ = integrate.quad(lambda chi: 1/deriv_chi_t(t_start, psi0, chi, p0, e0, a,  E0, L0, Q0),0,chi0)
    return result

def I_phi(t_start, psi0, chi0, phi0, p0, e0, a, E0, L0, Q0):
    result, _ = integrate.quad(lambda phi: 1/deriv_phi_t(t_start, psi0, chi0, p0, e0, a,  E0, L0, Q0),0,phi0)
    return result
def Phi_phi0_r0_theta0_to_psi0_chi0_phi0(Phi_phi0, Phi_theta0, Phi_r0, p0, e0, a, E0, L0, Q0, t_start = 0, max_iterations = 100, tol = 1e-6,
                                                psi0 = None, chi0 = None, phi0 = None):
    # Iterative solver
    if psi0 == None and chi0 == None and phi0 == None:
        psi0 = Phi_r0
        chi0 = Phi_theta0
        phi0 = Phi_phi0
    
    for iteration in range(max_iterations):
        print("Iteration", iteration)
        # Compute the total integrals based on current guesses
        I_psi_total, _ = integrate.quad(lambda psi: 1/deriv_psi_t(t_start, psi, chi0, p0, e0, a,  E0, L0, Q0),0,np.pi)
        I_chi_total, _ = integrate.quad(lambda chi: 1/deriv_chi_t(t_start, psi0, chi, p0, e0, a,  E0, L0, Q0),0,np.pi)
        I_phi_total, _ = integrate.quad(lambda phi: 1/deriv_phi_t(t_start, psi0, chi0, p0, e0, a,  E0, L0, Q0),0,np.pi)

        # Target values
        I_psi_target = (Phi_r0 / np.pi) * I_psi_total
        I_chi_target = (Phi_theta0 / np.pi) * I_chi_total
        I_phi_target = (Phi_phi0 / np.pi) * I_phi_total

        # Compute the function differences
        func_psi0 = I_psi(t_start, psi0, chi0, p0, e0, a, E0, L0, Q0) - I_psi_target
        func_chi0 = I_chi(t_start, psi0, chi0, p0, e0, a, E0, L0, Q0) - I_chi_target
        func_phi0 = I_phi(t_start, psi0, chi0, phi0, p0, e0, a, E0, L0, Q0) - I_phi_target

        # Update guesses using a simple method, like Newton-Raphson
        psi0_new = psi0 - func_psi0 / (I_psi_total / np.pi)  # Derivative approximation
        chi0_new = chi0 - func_chi0 / (I_chi_total / np.pi)  # Derivative approximation
        phi0_new = phi0 - func_phi0 / (I_phi_total / np.pi)  # Derivative approximation

        # Check for convergence
        if (np.abs(psi0_new - psi0) < tol and
            np.abs(chi0_new - chi0) < tol and
            np.abs(phi0_new - phi0) < tol):
            break

        # Update variables
        psi0 = psi0_new
        chi0 = chi0_new
        phi0 = phi0_new

    print(f"Converged after {iteration + 1} iterations.")
    print(f"psi_0 = {psi0}, chi_0 = {chi0}, phi_0 = {phi0}")

    return psi0, chi0, phi0