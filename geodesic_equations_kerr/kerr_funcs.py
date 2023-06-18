import numpy as np

def radial_potential(r, a, E, L, Q):
    """
    Calculate the generic Kerr radial potential for a given set of parameters.

    Parameters:
    - Radial coordinate r (float): Radial distance.
    - Primary spin parameter a (float): Parameter 'a'.
    - Geodesic Energy E (float): Parameter 'E'.
    - Geodesic azimuthal angular momentum L (float): Parameter 'L'.
    - Geodesic carter constant Q (float): Parameter 'Q'.

    Returns:
    float: The value of the radial potential.
    """
    c0 = (E**2 - 1)
    c1 = 2.0
    c2 = (a**2 * (E**2 - 1.0) - L**2 - Q)
    c3 = 2.0 * (Q + (a * E - L)**2)
    c4 = -Q * a**2
    return c0 * r**4 + c1 * r**3 + c2 * r**2 + c3 * r + c4

def roots_radial_potential(a, E, L, Q):
    """
    Find the roots of the radial potential equation for a given set of parameters.

    Parameters:
    - a (float): Spin Parameter 'a'.
    - E (float): Energy Parameter 'E'.
    - L (float): Azimuthal angular momentum Parameter 'L'.
    - Q (float): Carter constant Q Parameter 'Q'.

    Returns:
    tuple: A tuple containing the four roots (r_a, r_p, r_3, r_4) of the radial potential equation.

    Example:
    """

    # Coefficients
    c0 = (E**2 - 1)
    c1 = 2.0
    c2 = (a**2 * (E**2 - 1.0) - L**2 - Q)
    c3 = 2.0 * (Q + (a * E - L)**2)
    c4 = -Q * a**2

    coefficients = [c0, c1, c2, c3, c4]
    
    # Compute roots
    roots = np.roots(coefficients)
    r_a = roots[0]
    r_p = roots[1]
    r_3 = roots[2]
    r_4 = roots[3]
    return r_a, r_p, r_3, r_4

def theta_potential(theta,a, E, L, Q):
    """
    Find the roots of the z equation for a given set of parameters.

    Parameters:
    - theta (float): Parameter 'theta'
    - a (float): Parameter 'a'.
    - E (float): Parameter 'E'.
    - L (float): Parameter 'L'.
    - Q (float): Parameter 'Q'.

    Returns:
    float: The value of the polar potential

    """
    z = np.cos(theta)**2
    beta = a**2 * (1 - E**2)

    c0 = beta
    c1 = -(Q + L**2 + a**2 * (1 - E**2))
    c2 = Q

    return c0*z**2 + c1*z + c2

def roots_z_equation(a, E, L, Q):
    """
    Find the roots of the z equation for a given set of parameters.

    Parameters:
    - a (float): Parameter 'a'.
    - E (float): Parameter 'E'.
    - L (float): Parameter 'L'.
    - Q (float): Parameter 'Q'.

    Returns:
    tuple: A tuple containing the two roots (z_p, z_m) of the z equation.

    Example:
    >>> roots_z_equation(1.0, 0.5, 0.2, 0.1)
    (0.123+0j, 0.456+0j)
    """
    beta = a**2 * (1 - E**2)

    c0 = beta
    c1 = -(Q + L**2 + a**2 * (1 - E**2))
    c2 = Q

    coefficients = [c0, c1, c2]
    roots = np.roots(coefficients)
    z_p = max(roots)
    z_m = min(roots)
    return z_p, z_m

def z_chi(z_m,chi):
    return z_m * np.cos(chi)**2

def phi_potential(t,psi,chi, p,e,a, E, L, Q):
    """
    Calculates the phi potential function for a Kerr metric.

    Parameters:
    - t (float): Time value.
    - psi (float): Psi coordinate value.
    - chi (float): Chi coordinate value.
    - p (float): Radial coordinate parameter.
    - e (float): Eccentricity parameter.
    - a (float): Kerr metric parameter.
    - E (float): Energy parameter.
    - L (float): Angular momentum parameter.
    - Q (float): Carter constant parameter.

    Returns:
    - float: The value of the phi potential function.

    This function calculates the phi potential function, which is specific to the Kerr metric. It takes several parameters
    that define the spacetime and the coordinates of interest. The phi potential function is derived from the Kerr metric
    and is used to describe the behavior of particles in the spacetime.

    The phi potential function is calculated based on the given input parameters using various intermediate calculations,
    including the computation of radial and angular coordinates, determination of roots, and other relevant quantities.
    The resulting value represents the phi potential at the specified spacetime coordinates.

    Note: This function relies on the `roots_z_equation` function to compute the roots of the z equation, which are
    necessary for the phi potential calculation.

    """
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    _, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    sintheta_sqr = 1 - z_chi(z_m,chi)
    first_term = L/sintheta_sqr - a*E
    second_term = (a/delta)*(E * (r**2 + a**2) - L*a)
    return first_term + second_term 

def t_potential(t,psi,chi, p,e,a, E, L, Q):
    """
    Calculates the t potential function for a Kerr metric.

    Parameters:
    - t (float): Time value.
    - psi (float): Psi coordinate value.
    - chi (float): Chi coordinate value.
    - p (float): Radial coordinate parameter.
    - e (float): Eccentricity parameter.
    - a (float): Kerr metric parameter.
    - E (float): Energy parameter.
    - L (float): Angular momentum parameter.
    - Q (float): Carter constant parameter.

    Returns:
    - float: The value of the t potential function.

    This function calculates the t potential function, which is specific to the Kerr metric. It takes several parameters
    that define the spacetime and the coordinates of interest. The t potential function is derived from the Kerr metric
    and is used to describe the behavior of particles in the spacetime.

    The t potential function is calculated based on the given input parameters using various intermediate calculations,
    including the computation of radial and angular coordinates, determination of roots, and other relevant quantities.
    The resulting value represents the t potential at the specified spacetime coordinates.

    Note: This function relies on the `roots_z_equation` and `z_chi` functions to compute the roots of the z equation
    and z_chi value, respectively, which are necessary for the t potential calculation.
    """
    r = p/(1 + e*np.cos(psi)) # Radial coordinate - correct
    _, z_m = roots_z_equation(a, E, L, Q) # Roots of z equation
    delta = r**2 - 2*r + a**2 # Zeros of gtt and grr - correct
    sintheta_sqr = 1 - z_chi(z_m,chi)
    first_term = a*(L - a*E*sintheta_sqr)
    second_term = ((r**2 + a**2)/delta) * (E * (r**2 + a**2) - L*a)
    return first_term + second_term 


def deriv_psi_t(t,psi,chi, p,e,a, E, L, Q):
    """
    Calculates the derivative of psi with respect to t for a Kerr metric.

    Parameters:
    - t (float): Time value.
    - psi (float): Psi coordinate value.
    - chi (float): Chi coordinate value.
    - p (float): Radial coordinate parameter.
    - e (float): Eccentricity parameter.
    - a (float): Kerr metric parameter.
    - E (float): Energy parameter.
    - L (float): Angular momentum parameter.
    - Q (float): Carter constant parameter.

    Returns:
    - float: The derivative of psi with respect to t.

    This function calculates the derivative of psi with respect to t for a Kerr metric. It takes several parameters that
    define the spacetime and the coordinates of interest. The derivative of psi with respect to t is derived from the
    Kerr metric and represents the change in psi coordinate with respect to time.

    The derivative is calculated based on the given input parameters using various intermediate calculations, including
    the computation of radial coordinates, determination of roots, and other relevant quantities. The resulting value
    represents the rate of change of psi with respect to t at the specified spacetime coordinates.

    Note: This function relies on the `roots_radial_potential`, `roots_z_equation`, and `z_chi` functions to compute
    the roots of the radial potential, roots of the z equation, and z_chi value, respectively, which are necessary for
    the derivative calculation.
    """
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
    Calculates the derivative of chi with respect to t for a Kerr metric.

    Parameters:
    - t (float): Time value.
    - psi (float): Psi coordinate value.
    - chi (float): Chi coordinate value.
    - p (float): Radial coordinate parameter.
    - e (float): Eccentricity parameter.
    - a (float): Kerr metric parameter.
    - E (float): Energy parameter.
    - L (float): Angular momentum parameter.
    - Q (float): Carter constant parameter.

    Returns:
    - float: The derivative of chi with respect to t.

    This function calculates the derivative of chi with respect to t for a Kerr metric. It takes several parameters that
    define the spacetime and the coordinates of interest. The derivative of chi with respect to t is derived from the
    Kerr metric and represents the change in chi coordinate with respect to time.

    The derivative is calculated based on the given input parameters using various intermediate calculations, including
    the computation of roots, radial coordinates, and other relevant quantities. The resulting value represents the rate
    of change of chi with respect to t at the specified spacetime coordinates.

    Note: This function relies on the `roots_z_equation` and `z_chi` functions to compute the roots of the z equation
    and the z_chi value, respectively, which are necessary for the derivative calculation.
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
    """
    Calculates the derivative of phi with respect to t for a Kerr metric.

    Parameters:
    - t (float): Time value.
    - psi (float): Psi coordinate value.
    - chi (float): Chi coordinate value.
    - p (float): Radial coordinate parameter.
    - e (float): Eccentricity parameter.
    - a (float): Kerr metric parameter.
    - E (float): Energy parameter.
    - L (float): Angular momentum parameter.
    - Q (float): Carter constant parameter.

    Returns:
    - float: The derivative of phi with respect to t.

    This function calculates the derivative of phi with respect to t for a Kerr metric. It takes several parameters that
    define the spacetime and the coordinates of interest. The derivative of phi with respect to t is derived from the
    Kerr metric and represents the change in phi coordinate with respect to time.

    The derivative is calculated based on the given input parameters using the `phi_potential` and `t_potential`
    functions, which compute the phi potential and t potential, respectively. These potentials are used to determine the
    ratio of V_phi to V_t, which represents the rate of change of phi with respect to t at the specified spacetime
    coordinates.

    Note: This function relies on the `phi_potential` and `t_potential` functions to compute the phi potential and t
    potential, which are necessary for the derivative calculation.
    """

    V_phi = phi_potential(t,psi,chi, p,e,a, E, L, Q)
    V_t = t_potential(t,psi,chi, p,e,a, E, L, Q)
    

    return (V_phi/V_t)

