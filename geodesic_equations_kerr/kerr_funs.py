import numpy as np

def radial_potential(r,a,E,L,Q):
    c0 = (E**2 - 1)
    c1 = 2.0
    c2 = (a**2 * (E**2 - 1.0) - L**2 - Q)
    c3 = 2.0*(Q + (a*E - L)**2)
    c4 = -Q*a**2
    return c0*r**4 + c1*r**3 + c2 * r**2 + c3 * r + c4
def roots_radial_potential(a,E,L,Q):

    c0 = (E**2 - 1)
    c1 = 2.0
    c2 = (a**2 * (E**2 - 1.0) - L**2 - Q)
    c3 = 2.0*(Q + (a*E - L)**2)
    c4 = -Q*a**2
    
    coefficients = [c0, c1, c2, c3, c4]
    roots = np.roots(coefficients)
    r_a = roots[0]
    r_p = roots[1]
    r_3 = roots[2]
    r_4 = roots[3]
    return r_a, r_p, r_3, r_4

def roots_z_equation(a, E,L,Q):
    beta = a**2 * (1 - E**2)

    c0 = beta
    c1 = -(Q + L**2 + a**2 * (1 - E**2))
    c2 = Q

    coefficients = [c0,c1,c2]
    roots = np.roots(coefficients)
    z_p = max(roots)
    z_m = min(roots)
    return z_p, z_m 

def z_chi(z_m,chi):
    return z_m * np.cos(chi)**2

def phi_potential(t,psi,chi, p,e,a, E, L, Q):
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    _, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    sintheta_sqr = 1 - z_chi(z_m,chi)**2
    first_term = L/sintheta_sqr - a*E
    second_term = (a/delta)*(E * (r**2 + a**2) - L*a)
    return first_term + second_term 

def t_potential(t,psi,chi, p,e,a, E, L, Q):
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    _, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    sintheta_sqr = 1 - z_chi(z_m,chi)**2
    first_term = a*(L - a*E*sintheta_sqr)
    second_term = ((r**2 + a**2)/delta) * (E * (r**2 + a**2) - L*a)
    return first_term + second_term 


def deriv_psi_t(t,psi,chi, p,e,a, E, L, Q):
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    gam = E*(((r**2 + a**2)**2)/delta - a**2) - 2*r*a*L/delta # gamma value - correct

    _, _, r_3, r_4 = roots_radial_potential(a,E,L, Q)   # Define roots of radial potential
    _, z_m = roots_z_equation(a, E, L, Q)               # Define roots of 

    first_term_num = np.sqrt(1 - E**2)* ((p - r_3*(1-e)) - e * (p + r_3*(1 - e)*np.cos(psi)))**(1/2) # Looks OK
    second_term_num =  np.sqrt( (p - r_4*(1+e)) + e*(p - r_4*(1 + e)*np.cos(psi))) # Looks OK 
    denom = (1 - e**2) * (gam + a**2 * E * z_chi(z_m,chi))  # 

    return first_term_num * second_term_num / denom
def deriv_chi_t(t,psi,chi, p,e,a, E, L, Q):
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

def deriv_phi_t(t,psi,chi, p,e,a, E, L, Q):
    
    V_phi = phi_potential(t,psi,chi, p,e,a, E, L, Q)
    V_t = t_potential(t,psi,chi, p,e,a, E, L, Q)
    

    return (V_phi/V_t)

