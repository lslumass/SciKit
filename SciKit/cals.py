"""
minor functions for quick calculations and conversions
"""


import numpy as np

# constants
NA = 6.02214076e23  # Avogadro's number
kB = 1.380649e-23  # Boltzmann constant in J/K
R = 8.314462618  # Gas constant in J/(mol*K)


def cal_csat(number, box_length, droplet_radius=0):
    """
    calculate the concentration of dilute phase in a box with a droplet
    V_dilute = box_length^3 - 4/3 * pi * droplet_radius^3
    csat = number / V_dilute/NA*1e30    in uM
    """

    V_dilute = box_length**3 - 4 / 3 * np.pi * droplet_radius**3
    csat = number / V_dilute / NA * 1e30
    
    return csat


# =============================================================================
#  fit generalized Gaussian coil form factor (Hammouda) to get Rg and v
# =============================================================================
def lower_incomplete_gamma(s, x):
    """
    lower incomplete gamma function
    """
    from scipy.special import gammainc, gamma as gamma_func
    return gammainc(s, x) * gamma_func(s)

def Pq(q, Rg, nu):
    """Excluded-volume form factor P(q)."""
    U = (q**2 * Rg**2 * (2*nu + 1) * (2*nu + 2)) / 6.0

    a1 = 1.0 / (2.0 * nu)
    a2 = 1.0 / nu

    result = np.ones_like(q, dtype=np.float64)
    mask = U > 1e-12
    U_m = U[mask]

    term1 = (1.0 / (nu * U_m**a1)) * lower_incomplete_gamma(a1, U_m)
    term2 = (1.0 / (nu * U_m**a2)) * lower_incomplete_gamma(a2, U_m)
    result[mask] = term1 - term2

    return result

def polymer_excluded_volume(q, Wq, qmin=0.0, qmax=np.inf, Rg_guess=10.0, nu_guess=0.5, maxfev=10000):
    """
    Fit the excluded-volume form factor P(q) to the given data W(q).
    Parameters:
    q : array-like
        The scattering vector magnitudes.
    Wq : array-like
        The measured scattering intensities corresponding to q.
    qmin : float, optional
        Minimum q value to consider for fitting (default is 0.0).
    qmax : float, optional
        Maximum q value to consider for fitting (default is infinity).
    Rg_guess : float, optional
        Initial guess for the radius of gyration (default is 10.0).
    nu_guess : float, optional
        Initial guess for the Flory exponent (default is 0.5).
    Returns:
    Rg_fit : float
        Fitted radius of gyration.
    nu_fit : float
        Fitted Flory exponent.
    """
    from scipy.optimize import curve_fit
    
    q = np.asarray(q)
    Wq = np.asarray(Wq)
    # Filter data based on qmin and qmax
    mask = (q >= qmin) & (q <= qmax)
    q_m = q[mask]
    Wq_m = Wq[mask]

    # Fit the model to the data
    popt, pcov = curve_fit(Pq, q_m, Wq_m, p0=[Rg_guess, nu_guess], bounds=([0, 0.3], [np.inf, 1]), maxfev=maxfev)
    Rg, nu = popt
    Rg_err, nu_err = np.sqrt(np.diag(pcov))
    Wq_fit = Pq(q_m, *popt)
    chi2 = np.sum((Wq_m - Wq_fit) ** 2 / (len(Wq_m) - 2))

    return {"Rg": Rg, "nu":nu, "Rg_err": Rg_err, "nu_err": nu_err, "chi2": chi2, "q_fit": q_m, "Wq_fit": Wq_fit}