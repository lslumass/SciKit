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
    Calculate the saturation concentration of the dilute phase in a simulation
    box containing a spherical droplet.

    The dilute-phase volume is the total box volume minus the droplet volume:
        V_dilute = box_length^3 - (4/3) * pi * droplet_radius^3

    Parameters
    ----------
    number : int or float
        Number of molecules in the dilute phase.
    box_length : float
        Side length of the cubic simulation box in Angstroms.
    droplet_radius : float, optional
        Radius of the spherical droplet in Angstroms. Default is 0 (no droplet).

    Returns
    -------
    csat : float
        Saturation concentration of the dilute phase in micromolar (uM).
    """

    V_dilute = box_length**3 - 4 / 3 * np.pi * droplet_radius**3
    csat = number / V_dilute / NA * 1e30
    
    return csat


# =============================================================================
#  fit generalized Gaussian coil form factor (Hammouda) to get Rg and v
# =============================================================================
def lower_incomplete_gamma(s, x):
    """
    Evaluate the lower incomplete gamma function gamma(s, x).

    Parameters
    ----------
    s : float
        Shape parameter (upper integration limit exponent).
    x : float or array-like
        Upper limit of integration.

    Returns
    -------
    float or array-like
        Value of the lower incomplete gamma function gamma(s, x).
    """
    from scipy.special import gammainc, gamma as gamma_func
    return gammainc(s, x) * gamma_func(s)

def Pq(q, Rg, nu):
    """
    Compute the excluded-volume (generalized Gaussian coil) form factor P(q)
    following the Hammouda model.

    The form factor is expressed in terms of the lower incomplete gamma function
    and depends on the radius of gyration Rg and the Flory exponent nu:
        U = q^2 * Rg^2 * (2*nu + 1) * (2*nu + 2) / 6

    Parameters
    ----------
    q : array-like
        Scattering vector magnitudes (1/Angstrom).
    Rg : float
        Radius of gyration of the polymer chain (Angstrom).
    nu : float
        Flory exponent (0 < nu <= 1); nu = 0.5 for ideal chain,
        nu ~ 0.588 for self-avoiding walk in good solvent.

    Returns
    -------
    result : ndarray
        Form factor P(q), normalized so that P(0) = 1.
    """
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
    Fit the excluded-volume form factor P(q) to measured scattering data W(q)
    to extract the radius of gyration Rg and the Flory exponent nu.

    Parameters
    ----------
    q : array-like
        Scattering vector magnitudes (1/Angstrom).
    Wq : array-like
        Measured scattering intensities corresponding to q, normalized so
        that W(0) ~ 1.
    qmin : float, optional
        Lower bound of the q range used for fitting. Default is 0.0.
    qmax : float, optional
        Upper bound of the q range used for fitting. Default is infinity.
    Rg_guess : float, optional
        Initial guess for the radius of gyration in Angstroms. Default is 10.0.
    nu_guess : float, optional
        Initial guess for the Flory exponent. Default is 0.5.
    maxfev : int, optional
        Maximum number of function evaluations for the curve fitting. Default
        is 10000.

    Returns
    -------
    dict with keys:
        Rg : float
            Fitted radius of gyration (Angstrom).
        nu : float
            Fitted Flory exponent.
        Rg_err : float
            Standard error of the fitted Rg.
        nu_err : float
            Standard error of the fitted nu.
        chi2 : float
            Reduced chi-squared statistic of the fit.
        q_fit : ndarray
            The subset of q values used in the fit.
        Wq_fit : ndarray
            The fitted P(q) values evaluated over q_fit.
    """
    from scipy.optimize import curve_fit
    
    q = np.asarray(q)
    Wq = np.asarray(Wq)
    # Filter data based on qmin and qmax
    mask = (q >= qmin) & (q <= qmax)
    q_m = q[mask]
    Wq_m = Wq[mask]

    # Fit the model to the data
    popt, pcov = curve_fit(Pq, q_m, Wq_m, p0=[Rg_guess, nu_guess], bounds=([0, 0.0], [np.inf, 1]), maxfev=maxfev)
    Rg, nu = popt
    Rg_err, nu_err = np.sqrt(np.diag(pcov))
    Wq_fit = Pq(q_m, *popt)
    chi2 = np.sum((Wq_m - Wq_fit) ** 2 / (len(Wq_m) - 2))

    return {"Rg": Rg, "nu":nu, "Rg_err": Rg_err, "nu_err": nu_err, "chi2": chi2, "q_fit": q_m, "Wq_fit": Wq_fit}


# =============================================================================
#  find the first linear region in Wq-q plot to get v
# =============================================================================

def intraCF2nu(q, Wq, qmax=1.65, window=55):
    """
    Estimate the Flory exponent nu from the intrachain structure factor W(q)
    by identifying the first linear (power-law) scaling region in a log-log
    plot of W(q) vs q.

    The function first fits the full excluded-volume form factor to determine
    Rg, then restricts the analysis to the Porod regime 1/Rg < q < qmax.
    A sliding window of fixed size is used to fit local power-law slopes;
    two independent methods are applied to locate the best-fit linear region:

      - Method 1 (nu1): the window with the first local maximum in R².
      - Method 2 (nu2): the window with the first local minimum in slope
        (most negative power-law exponent).

    In each case, nu = -1 / slope.

    Parameters
    ----------
    q : array-like
        Scattering vector magnitudes (1/Angstrom).
    Wq : array-like
        Intrachain structure factor values corresponding to q.
    qmax : float, optional
        Upper bound of the q range used for slope analysis. Default is 1.65.
    window : int, optional
        Number of consecutive q points used in each sliding-window power-law
        fit. Default is 55.

    Returns
    -------
    dict with keys:
        nu1 : float
            Flory exponent from the first local R² maximum method.
        nu2 : float
            Flory exponent from the first local slope minimum method.
        slopes : list of float
            Power-law slopes from each sliding window position.
        qs : list of float
            Starting q value of each sliding window.
        R2 : list of float
            Coefficient of determination (R²) for each sliding window fit.
    """
    results = polymer_excluded_volume(q, Wq, qmax=qmax)
    Rg = results["Rg"]
    qmin = 1.0 / Rg

    from scipy.optimize import curve_fit

    def fit_func(q, a, b):
        return np.power(10, a * np.log10(q) + b)

    # Find index where q is qmin and qmax
    lower = next(i for i, q_val in enumerate(q) if q_val >= qmin)
    upper = next(i for i, q_val in enumerate(q) if q_val > qmax)

    slopes = []
    qs = []
    r_squareds = []

    for i in range(lower, upper):
        if i + window > len(q):
            break
        xs, ys = q[i:i+window], Wq[i:i+window]
        popt, pcov = curve_fit(fit_func, xs, ys, p0=[-2, -0.5])

        # Compute R²
        y_fit = fit_func(xs, *popt)
        ss_res = np.sum((ys - y_fit) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)

        slopes.append(popt[0])
        qs.append(q[i])
        r_squareds.append(r2)

    # Method 1: First local maximum in R²
    first_r2_max_idx = None
    for i in range(1, len(r_squareds) - 1):
        if r_squareds[i] > r_squareds[i-1] and r_squareds[i] > r_squareds[i+1]:
            first_r2_max_idx = i
            break
    if first_r2_max_idx is None:
        first_r2_max_idx = np.argmax(r_squareds)

    nu1 = -1.0 / slopes[first_r2_max_idx]

    # Method 2: First local minimum in slopes
    first_slope_min_idx = None
    for i in range(1, len(slopes) - 1):
        if slopes[i] < slopes[i-1] and slopes[i] < slopes[i+1]:
            first_slope_min_idx = i
            break
    if first_slope_min_idx is None:
        first_slope_min_idx = np.argmin(slopes)

    nu2 = -1.0 / slopes[first_slope_min_idx]

    return {
        "nu1": nu1,
        "nu2": nu2,
        "slopes": slopes,
        "qs": qs,
        "R2": r_squareds
    }


# =============================================================================
#  fit the density distribution to "hyperbolic tangent function" to get the droplet radius
#   J. S. Rowlinson and B. Widom , Molecular Theory of Capillarity , Oxford University, Oxford, 1982
# =============================================================================

def droplet(r, c):
    """
    Fit a radial density profile to a hyperbolic tangent interface model to
    extract droplet properties such as the radius and interfacial width.

    The model follows Rowlinson & Widom (1982) for a planar-like interface
    mapped onto a spherical geometry:
        c(r) = (cin + cout)/2 - (cin - cout)/2 * tanh((r - r0) / d)

    where cin and cout are the interior and exterior densities, r0 is the
    equimolar dividing surface radius, and d is the interfacial width.

    Parameters
    ----------
    r : array-like
        Radial distances from the droplet centre (Angstrom).
    c : array-like
        Density (or concentration) values at each radial position.

    Returns
    -------
    dict with keys:
        in : float
            Fitted density inside the droplet (cin).
        out : float
            Fitted density outside the droplet (cout).
        r0 : float
            Fitted equimolar radius of the droplet (Angstrom).
        d : float
            Fitted interfacial width (Angstrom); smaller values indicate a
            sharper interface.
        x_fit : ndarray
            Uniformly spaced radial values over [0.9*r.min(), 1.1*r.max()]
            used to evaluate the fitted curve.
        y_fit : ndarray
            Fitted density profile evaluated at x_fit.
    """
    def func(r, cin, cout, r0, d):
        return (cin + cout) / 2 - (cin - cout) / 2 * np.tanh((r - r0) / d)
    
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, r, c, p0=[np.max(c), np.min(c), np.mean(r), 1.0])
    cin, cout, r0, d = popt
    x_fit = np.linspace(r.min()*0.9, r.max()*1.1, 200)
    y_fit = func(x_fit, *popt)
    return {"in": cin, "out": cout, "r0": r0, "d": d, "x_fit": x_fit, "y_fit": y_fit}


# =============================================================================
#  general fit function for fitting to get the fitted parameters and fitted curve with confidence interval
# =============================================================================

def general_fit(func, xs, ys, yerrs=None, xrange=None, p0=None, bounds=(-np.inf, np.inf), maxfev=10000):
    """
    Fit an arbitrary model function to data with uncertainties and compute the
    fitted curve along with a propagated confidence interval.

    The confidence band is obtained via numerical Jacobian propagation:
        var(y_fit) = J @ cov @ J^T  (diagonal elements)
    where J is the Jacobian of func with respect to the fit parameters,
    evaluated at each point in xfit.

    Parameters
    ----------
    func : callable
        Model function with signature func(x, *params) -> array-like.
    xs : array-like
        Independent variable data points.
    ys : array-like
        Dependent variable data points.
    yerrs : array-like, optional
        Absolute uncertainties (standard deviations) on ys, used as weights
        in the chi-squared minimization. Default is None, in which case
        all points are weighted equally (unweighted least-squares fit) and
        absolute_sigma is set to False.
    xrange : tuple of (float, float), optional
        (x_min, x_max) range over which to evaluate the fitted curve.
        Default is None, in which case the range of xs is used.
    p0 : array-like, optional
        Initial guesses for the fit parameters. Passed directly to
        scipy.optimize.curve_fit. Default is None (uses curve_fit defaults).
    bounds : 2-tuple of array-like, optional
        Lower and upper bounds on the fit parameters. Default is
        (-inf, inf) (unbounded).
    maxfev : int, optional
        Maximum number of function evaluations allowed. Default is 10000.

    Returns
    -------
    dict with keys:
        params : ndarray
            Best-fit parameter values.
        perr : ndarray
            Standard errors of the fitted parameters (sqrt of covariance
            diagonal).
        xfit : ndarray
            10000 evenly spaced x values spanning xrange.
        yfit : ndarray
            Model evaluated at xfit with the best-fit parameters.
        yfit_err : ndarray
            1-sigma confidence band on yfit from error propagation.
    """
    from scipy.optimize import curve_fit

    xs = np.asarray(xs)

    if xrange is None:
        xrange = (xs.min(), xs.max())

    absolute_sigma = yerrs is not None

    params, cov = curve_fit(
        func, xs, ys, sigma=yerrs, absolute_sigma=absolute_sigma,
        p0=p0, bounds=bounds, maxfev=maxfev
    )
    perr = np.sqrt(np.diag(cov))

    xfit = np.linspace(xrange[0], xrange[1], 10000)
    yfit = func(xfit, *params)

    # Numerically compute the Jacobian: df/dp_i for each parameter
    dp = perr * 1e-4  # small step relative to each param's uncertainty
    dp = np.where(dp == 0, 1e-8, dp)  # fallback if perr is zero

    jacobian = np.zeros((len(xfit), len(params)))
    for i, (p, h) in enumerate(zip(params, dp)):
        params_up = params.copy()
        params_up[i] += h
        params_dn = params.copy()
        params_dn[i] -= h
        jacobian[:, i] = (func(xfit, *params_up) - func(xfit, *params_dn)) / (2 * h)

    # Error propagation: yfit_err^2 = J @ cov @ J^T (diagonal elements only)
    yfit_var = np.einsum("ij,jk,ik->i", jacobian, cov, jacobian)
    yfit_err = np.sqrt(yfit_var)

    return {"params": params, "perr": perr, "xfit": xfit, "yfit": yfit, "yfit_err": yfit_err}