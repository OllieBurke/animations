import numpy as np

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

def radial_potential(r,a,E,L,Q):
    c0 = (E**2 - 1)
    c1 = 2.0
    c2 = (a**2 * (E**2 - 1.0) - L**2 - Q)
    c3 = 2.0*(Q + (a*E - L)**2)
    c4 = -Q*a**2
    return c0*r**4 + c1*r**3 + c2 * r**2 + c3 * r + c4


