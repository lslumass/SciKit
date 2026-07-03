"""
Analysis utilities for molecular dynamics simulation post-processing.

Provides functions for statistical analysis, histogram/distribution conversion,
PCA-based free-energy landscapes, block averaging, and data loading helpers
built on top of NumPy, SciPy, and MDAnalysis.
"""

import numpy as np
import MDAnalysis as mda


def scatter2hist(point_list, num_bin, styles):
    """
    Convert a 1-D array of scatter points to a histogram or smooth distribution.

    Computes a normalized histogram from raw data and returns either a
    stepped distribution (suitable for ``ax.plot`` step-style rendering) or a
    cubic/quadratic spline-smoothed curve, depending on ``styles``.
    Zero-valued padding is appended to both ends so the distribution reaches
    zero at the boundaries.

    Parameters
    ----------
    point_list : array_like
        1-D sequence of scalar data points to histogram.
    num_bin : int
        Number of histogram bins. For smoothed styles, the output x-grid
        is upsampled to ``num_bin * 10`` points.
    styles : {'pdf', 'pmf', 's_pdf', 's_pmf'}
        Output type:

        - ``'pdf'``  : Stepped probability density function
          (area under curve ≈ 1).
        - ``'pmf'``  : Stepped probability mass function
          (sum of values ≈ 1).
        - ``'s_pdf'``: Cubic spline-smoothed PDF curve.
        - ``'s_pmf'``: Quadratic spline-smoothed PMF curve.

    Returns
    -------
    x : numpy.ndarray
        Bin-center positions (or smoothed x-grid for ``'s_pdf'``/``'s_pmf'``),
        with one extra zero-padded point prepended and appended.
    y : numpy.ndarray
        Corresponding PDF or PMF values, zero-padded at both ends.

    Raises
    ------
    ValueError
        If ``styles`` is not one of the four accepted strings.

    Notes
    -----
    The smoothed variants use ``scipy.interpolate.make_interp_spline`` with
    spline order ``k=3`` (PDF) or ``k=2`` (PMF). Negative spline artefacts
    near the zero-padded boundaries are possible for small ``num_bin`` values.

    Examples
    --------
    >>> x, y = scatter2hist(distances, num_bin=50, styles='pdf')
    >>> ax.plot(x, y)

    >>> x_smooth, y_smooth = scatter2hist(distances, num_bin=50, styles='s_pdf')
    >>> ax.plot(x_smooth, y_smooth)
    """
    from scipy.interpolate import make_interp_spline

    counts, bins = np.histogram(point_list, num_bin)
    pmf = counts / counts.sum()
    pdf = pmf / np.diff(bins)

    # Use bin centers instead of edges
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_width = np.diff(bins)[0]

    # Pad edges for smooth or full histogram plotting
    def pad_edges(x, y):
        x = np.insert(x, 0, x[0] - bin_width)
        x = np.append(x, x[-1] + bin_width)
        y = np.insert(y, 0, 0)
        y = np.append(y, 0)
        return x, y

    if styles == 'pdf':
        bin_centers, pdf = pad_edges(bin_centers, pdf)
        return bin_centers, pdf

    elif styles == 'pmf':
        bin_centers, pmf = pad_edges(bin_centers, pmf)
        return bin_centers, pmf

    elif styles == 's_pdf':
        bin_centers, pdf = pad_edges(bin_centers, pdf)
        bins_smooth = np.linspace(bin_centers.min(), bin_centers.max(), num_bin * 10)
        spline = make_interp_spline(bin_centers, pdf, k=3)
        pdf_smooth = spline(bins_smooth)
        return bins_smooth, pdf_smooth

    elif styles == 's_pmf':
        bin_centers, pmf = pad_edges(bin_centers, pmf)
        bins_smooth = np.linspace(bin_centers.min(), bin_centers.max(), num_bin * 10)
        spline = make_interp_spline(bin_centers, pmf, k=2)
        pmf_smooth = spline(bins_smooth)
        return bins_smooth, pmf_smooth

    else:
        raise ValueError("styles must be one of ['pdf', 'pmf', 's_pdf', 's_pmf']")


def mda_pca(psf, dcd, sel, align=True):
    """
    Perform PCA on an MD trajectory and return the first two principal components.

    Loads a CHARMM PSF + DCD trajectory pair, selects atoms, and runs
    MDAnalysis PCA. Returns the per-frame projections onto PC1 and PC2
    along with their individual explained-variance fractions.

    Parameters
    ----------
    psf : str
        Path to the CHARMM PSF topology file.
    dcd : str
        Path to the DCD trajectory file.
    sel : str
        MDAnalysis atom-selection string applied to both alignment and PCA
        (e.g., ``'backbone'``, ``'name CA'``).
    align : bool, optional
        If ``True`` (default), align all frames to the first frame before
        computing the covariance matrix.

    Returns
    -------
    pc1 : numpy.ndarray, shape (n_frames,)
        Per-frame projection onto the first principal component.
    pc2 : numpy.ndarray, shape (n_frames,)
        Per-frame projection onto the second principal component.
    var1 : float
        Explained variance fraction of PC1 (value in [0, 1]).
    var2 : float
        Explained variance fraction of PC2 (value in [0, 1]).

    Notes
    -----
    ``var1`` is taken directly from ``pc.cumulated_variance[0]``.
    ``var2`` is the *incremental* variance of PC2, computed as
    ``cumulated_variance[1] - cumulated_variance[0]``.
    Multiply by 100 to obtain a percentage.

    Examples
    --------
    >>> pc1, pc2, var1, var2 = mda_pca('system.psf', 'traj.dcd', 'backbone')
    >>> print(f'PC1: {var1*100:.1f}%  PC2: {var2*100:.1f}%')
    """
    from MDAnalysis.analysis import pca

    u = mda.Universe(psf, dcd)
    atomgroup = u.select_atoms(sel)
    pc = pca.PCA(u, select=sel, align=align).run()
    pc1, pc2 = pc.transform(atomgroup)[:, 0], pc.transform(atomgroup)[:, 1]
    var1, var2 = pc.cumulated_variance[0], pc.cumulated_variance[1] - pc.cumulated_variance[0]
    return pc1, pc2, var1, var2


def pca2fe(pc1, pc2, num_bin=100):
    """
    Convert 2-D PCA projections to a free-energy landscape in units of k\ :sub:`B`\ T.

    Bins ``(pc1, pc2)`` pairs into a 2-D histogram, normalises to a PMF,
    and applies the Boltzmann inversion ``F = -ln(P)``. The global minimum
    of the finite free-energy values is shifted to zero.

    Parameters
    ----------
    pc1 : array_like, shape (n_frames,)
        Per-frame projection onto the first principal component.
    pc2 : array_like, shape (n_frames,)
        Per-frame projection onto the second principal component.
    num_bin : int, optional
        Number of bins along each axis of the 2-D histogram (default ``100``).

    Returns
    -------
    fe : numpy.ndarray, shape (num_bin, num_bin)
        Free-energy surface in units of k\ :sub:`B`\ T, with the global
        minimum set to 0. Bins with zero occupancy contain ``inf``.

    Notes
    -----
    The local minimum (highest-occupancy bin) coordinates in data space are
    computed internally but not currently returned. Retrieve them by
    inspecting ``xedges`` / ``yedges`` if needed in future extensions.

    Examples
    --------
    >>> fe = pca2fe(pc1, pc2, num_bin=80)
    >>> plt.imshow(fe.T, origin='lower')
    """
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=num_bin)
    pmf = H / np.sum(H)

    # convert pmf to free energy
    fe = -np.log(pmf)
    fe_min = np.min(fe[np.isfinite(fe)])
    fe -= fe_min

    # find the local minimum
    local_minimum = np.unravel_index(np.argmax(H), H.shape)
    localx, localy = xedges[local_minimum[0]], yedges[local_minimum[1]]
    return fe


def plt_pca(axs, fe, var1=None, var2=None, cmap='viridis', contour=False, levels=5):
    """
    Plot a 2-D free-energy landscape from PCA projections.

    Displays the free-energy array as a colour-mapped image with an
    attached colorbar. Axis tick marks are suppressed so that the plot
    reflects relative rather than absolute PC coordinates. Optionally
    overlays iso-energy contour lines.

    Parameters
    ----------
    axs : matplotlib.axes.Axes
        Target axes on which the landscape will be drawn.
    fe : numpy.ndarray, shape (N, N)
        Free-energy surface in units of k\ :sub:`B`\ T, as returned by
        :func:`pca2fe`. Transposed internally before display.
    var1 : float or None, optional
        Explained variance fraction of PC1 (value in [0, 1]). If provided
        together with ``var2``, percentage labels are added to the axis.
        Default ``None`` omits the percentage.
    var2 : float or None, optional
        Explained variance fraction of PC2 (value in [0, 1]).
        Default ``None`` omits the percentage.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the free-energy surface (default ``'viridis'``).
    contour : bool, optional
        If ``True``, overlay black iso-energy contour lines (default ``False``).
    levels : int, optional
        Number of contour levels drawn when ``contour=True`` (default ``5``).

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 6, 5)
    >>> fe = pca2fe(pc1, pc2)
    >>> plt_pca(axs, fe, var1=0.35, var2=0.12, contour=True, levels=8)
    """
    im = axs.imshow(fe.T, origin='lower', cmap=cmap)
    axs.figure.colorbar(im, ax=axs, label='Free energy (k$_B$T)', fraction=0.046)
    if var1 is None:
        axs.set(xlabel='PC1', xticks=[], ylabel='PC2', yticks=[])
    else:
        axs.set(xlabel=f'PC1 ({var1*100:.2f}%)', xticks=[], ylabel=f'PC2 ({var2*100:.2f}%)', yticks=[])

    if contour:
        axs.contour(fe.T, levels=levels, colors='k')


def pca2d(axs, psf, dcd, sel, num_bin=100, align=True, cmap='viridis'):
    """
    Compute and plot a 2-D PCA free-energy landscape in a single call.

    Convenience wrapper that combines :func:`mda_pca`, :func:`pca2fe`, and
    :func:`plt_pca` into one step. Loads the trajectory, runs PCA, converts
    to a free-energy surface, and renders the result directly onto ``axs``.
    Prefer this function for quick exploratory plots; use the individual
    functions when you need intermediate results (e.g., ``pc1``/``pc2`` for
    further analysis).

    Parameters
    ----------
    axs : matplotlib.axes.Axes
        Target axes on which the landscape will be drawn.
    psf : str
        Path to the CHARMM PSF topology file.
    dcd : str
        Path to the DCD trajectory file.
    sel : str
        MDAnalysis atom-selection string (e.g., ``'backbone'``).
    num_bin : int, optional
        Number of bins along each PCA axis (default ``100``).
    align : bool, optional
        If ``True`` (default), align frames before PCA.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the free-energy surface (default ``'viridis'``).

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 6, 5)
    >>> pca2d(axs, 'system.psf', 'traj.dcd', 'backbone', num_bin=80)
    """
    from MDAnalysis.analysis import pca

    u = mda.Universe(psf, dcd)
    atomgroup = u.select_atoms(sel)
    pc = pca.PCA(u, select=sel, align=align).run()
    pc1, pc2 = pc.transform(atomgroup)[:, 0], pc.transform(atomgroup)[:, 1]
    var1, var2 = pc.cumulated_variance[0], pc.cumulated_variance[1] - pc.cumulated_variance[0]
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=num_bin)
    pmf = H / np.sum(H)

    # convert pmf to free energy
    fe = -np.log(pmf)
    fe_min = np.min(fe[np.isfinite(fe)])
    fe -= fe_min

    im = axs.imshow(fe.T, origin='lower', cmap=cmap)
    axs.figure.colorbar(im, ax=axs, label='Free energy (k$_B$T)', fraction=0.046)
    axs.set(xlabel=f'PC1 ({var1*100:.2f}%)', xticks=[], ylabel=f'PC2 ({var2*100:.2f}%)', yticks=[])


def block_mean(data, division):
    """
    Estimate the mean and uncertainty of a time series using two-block averaging.

    Discards an initial fraction of the data as equilibration, then splits
    the remaining production region into two equal blocks. The reported
    average is the mean of the two block means and the error is their
    standard deviation — a simple variance estimator that accounts for
    serial correlation in MD observables.

    Parameters
    ----------
    data : array_like
        1-D sequence of scalar values ordered along simulation time.
    division : int
        Controls the equilibration cutoff: the first ``1/division`` of
        the data is discarded. The remaining ``(1 - 1/division)`` fraction
        is split into two equal blocks for the error estimate.
        For example, ``division=5`` discards the first 20 % of frames.

    Returns
    -------
    average : float
        Mean of the two block averages.
    error : float
        Standard deviation of the two block averages (half the difference
        between them), representing the statistical uncertainty.

    Examples
    --------
    >>> avg, err = block_mean(rg_timeseries, division=5)
    >>> print(f'Rg = {avg:.2f} ± {err:.2f} Å')
    """
    l = len(data)
    start_frame = int(l / division)
    half = start_frame + int((l - start_frame) / 2)
    part1 = np.mean(data[start_frame:half])
    part2 = np.mean(data[half:])
    average = np.mean([part1, part2])
    error = np.std([part1, part2])
    return average, error


def rms(data):
    """
    Compute the root-mean-square (RMS) value of an array.

    Parameters
    ----------
    data : array_like
        1-D or N-D array of numeric values.

    Returns
    -------
    float
        RMS value: ``sqrt(mean(data**2))``.

    Examples
    --------
    >>> rms([1, -2, 3, -4])
    2.7386...
    """
    data = np.asarray(data)
    return np.sqrt(np.mean(data**2))


def block_bootstrap(data, blocksize=1000, nsamples=1000, statistic=rms):
    """
    Estimate the standard deviation of a statistic via circular block bootstrap.

    Accounts for temporal autocorrelation in MD time series by resampling
    contiguous blocks rather than individual frames. Uses the *circular*
    (wraparound) variant to avoid boundary effects at the end of the array.

    Parameters
    ----------
    data : array_like
        1-D time-ordered array of data points.
    blocksize : int, optional
        Number of consecutive frames in each resampled block (default ``1000``).
        Should be larger than the autocorrelation time of the observable.
    nsamples : int, optional
        Number of independent bootstrap replicates to generate (default ``1000``).
    statistic : callable, optional
        Function applied to each bootstrap replicate to compute the quantity
        of interest. Must accept a 1-D array and return a scalar.
        Default is :func:`rms`; other useful choices include ``np.mean``
        or ``np.std``.

    Returns
    -------
    sd : float
        Bootstrap standard deviation of ``statistic`` across all replicates,
        computed with ``ddof=1``.

    Notes
    -----
    Each bootstrap replicate is built by drawing random start indices,
    extracting blocks of length ``blocksize`` with circular index wrapping
    (``index % n``), and concatenating until the sample reaches length ``n``.
    The final sample is trimmed to exactly ``n`` frames.

    Choosing ``blocksize`` is the main tuning parameter: too small underestimates
    the error; too large increases variance of the estimator. A common heuristic
    is to set it to roughly 5–10× the autocorrelation time.

    Examples
    --------
    >>> sd = block_bootstrap(rg_timeseries, blocksize=500, statistic=np.mean)
    >>> print(f'Bootstrap std of <Rg>: {sd:.4f} Å')
    """
    data = np.asarray(data)
    n = len(data)
    bootstrap_stats = []
    for _ in range(nsamples):
        # Generate bootstrap sample using circular block bootstrap
        bootstrap_sample = []
        while len(bootstrap_sample) < n:
            start_idx = np.random.randint(0, n)
            block = []
            for i in range(blocksize):
                block.append(data[(start_idx + i) % n])
            bootstrap_sample.extend(block)

        # trim to exact length
        bootstrap_sample = bootstrap_sample[:n]
        bootstrap_stats.append(statistic(bootstrap_sample))

    sd = np.std(bootstrap_stats, ddof=1)
    return sd


def remove_zero(xs, ys):
    """
    Find the x-range that spans the non-zero support of a distribution.

    Scans ``ys`` from both ends to locate the first and last non-zero values,
    then returns the corresponding ``xs`` positions offset by ±5 index
    positions to retain a small margin around the support.

    Parameters
    ----------
    xs : array_like
        1-D array of x-axis values (e.g., bin centers).
    ys : array_like
        1-D array of y-axis values (e.g., PDF or PMF). Must be the same
        length as ``xs``.

    Returns
    -------
    x_nozero : 1-D array
        x values within the non-zero support of the distribution.
    y_nozero : 1-D array
        Corresponding y values within the non-zero support.

    Examples
    --------
    >>> x, y = scatter2hist(distances, num_bin=50, styles='pdf')
    >>> x_new, y_new = remove_zero(x, y)
    >>> ax.plot(x_new, y_new)
    """
    for idx in range(len(xs)):
        if ys[idx] != 0:
            start = idx - 5
            break
    for idx in reversed(range(len(xs))):
        if ys[idx] != 0:
            end = idx + 5
            break
    return xs[start:end], ys[start:end]


def myload(filename, *args, **kwargs):
    """
    Load a whitespace-delimited text file and unpack each column as a separate array.

    Wraps ``numpy.loadtxt`` and returns each column as an individual element
    of a tuple, making it convenient to unpack multi-column data files with
    a single assignment. Up to eight trailing ``0`` sentinels are appended
    so that callers can safely unpack into more variables than there are
    columns without raising a ``ValueError``.

    Parameters
    ----------
    filename : str or path-like
        Path to the text file to load.
    *args
        Additional positional arguments forwarded to ``numpy.loadtxt``
        (e.g., ``skiprows``).
    **kwargs
        Additional keyword arguments forwarded to ``numpy.loadtxt``
        (e.g., ``delimiter``, ``usecols``, ``dtype``).

    Returns
    -------
    tuple
        A tuple whose first ``n`` elements are 1-D ``numpy.ndarray`` objects,
        one per column in the file, followed by eight integer zeros as padding.
        The total length is therefore ``n + 8``.

    Notes
    -----
    The eight trailing zeros allow patterns like::

        col1, col2, *_ = myload('data.txt')

    without the caller needing to know the exact column count. However, the
    padding means ``len(myload(...))`` is always ``n_cols + 8`` regardless of
    the actual file width.

    Examples
    --------
    >>> time, rg, energy, *_ = myload('observables.dat', skiprows=1)
    >>> plt.plot(time, rg)
    """
    data = np.loadtxt(filename, *args, **kwargs)
    cols = len(data[0, :])
    results = []
    for col in range(cols):
        results.append(data[:, col])
    results = results + [0, 0, 0, 0, 0, 0, 0, 0]
    return tuple(results)


def get_overlap(x1, y1, x2, y2):
    """
    Find the x values present in both datasets, and return the corresponding
    y values from each dataset at those shared x positions.

    No interpolation is performed — only x values that literally appear in
    both x1 and x2 are included in the output. The result is independent of
    argument order: get_overlap(x1, y1, x2, y2) and get_overlap(x2, y2, x1, y1)
    will return the same x_overlap (though y1_overlap and y2_overlap will swap).

    Parameters
    ----------
    x1 : array-like
        X-coordinates of the first dataset (assumed sorted ascending,
        no duplicates).
    y1 : array-like
        Y-values corresponding to x1.
    x2 : array-like
        X-coordinates of the second dataset (assumed sorted ascending,
        no duplicates).
    y2 : array-like
        Y-values corresponding to x2.

    Returns
    -------
    x_overlap : np.ndarray
        X-values present in both x1 and x2, sorted ascending.
    y1_overlap : np.ndarray
        Y-values from dataset 1 at x_overlap.
    y2_overlap : np.ndarray
        Y-values from dataset 2 at x_overlap.

    Raises
    ------
    ValueError
        If x1 and x2 share no common x values.

    Example
    -------
    >>> x1 = [1, 2, 3, 4, 5]
    >>> y1 = [10, 20, 30, 40, 50]
    >>> x2 = [3, 4, 5, 6, 7]
    >>> y2 = [300, 400, 500, 600, 700]
    >>> x, y1_out, y2_out = get_overlap(x1, y1, x2, y2)
    >>> # x = [3, 4, 5], y1_out = [30, 40, 50], y2_out = [300, 400, 500]
    """
    x1, y1, x2, y2 = map(np.asarray, (x1, y1, x2, y2))

    x_overlap = np.intersect1d(x1, x2)  # values present in both, order-independent

    if len(x_overlap) == 0:
        raise ValueError(
            f"No overlapping x values: x1 spans [{x1.min()}, {x1.max()}], "
            f"x2 spans [{x2.min()}, {x2.max()}]."
        )

    # Index directly — no interpolation needed since both arrays contain these x values
    y1_overlap = y1[np.isin(x1, x_overlap)]
    y2_overlap = y2[np.isin(x2, x_overlap)]

    return x_overlap, y1_overlap, y2_overlap


def stack_jagged(arrays, model="min"):
    """
    Stack a list of 1D arrays as columns, aligning on the overlap.

    Parameters
    ----------
    arrays : list of array-like
        Input arrays or lists, can have different lengths.
    model : {'min', 'max'}, optional
        'min' : trim all arrays to the shortest length (default).
        'max' : pad shorter arrays with NaN to match the longest length.

    Returns
    -------
    np.ndarray
        2D array of shape (min_len, n_arrays) if model='min',
        or (max_len, n_arrays) if model='max'.
    """
    arrays = [np.asarray(arr, dtype=float) for arr in arrays]

    if model == "min":
        target_len = min(len(arr) for arr in arrays)
        arrays = [arr[:target_len] for arr in arrays]

    elif model == "max":
        target_len = max(len(arr) for arr in arrays)
        arrays = [
            np.pad(arr, (0, target_len - len(arr)), constant_values=np.nan)
            for arr in arrays
        ]

    result = arrays[0]
    for arr in arrays[1:]:
        result = np.column_stack((result, arr))

    return result


def mylinregress(xs, ys, xrange=None):
    """
    Perform linear regression on (x, y) data, optionally restricted to a specific x-range.

    Parameters
    ----------
    xs : array-like
        Independent variable values.
    ys : array-like
        Dependent variable values, same length as xs.
    xrange : tuple of (float, float), optional
        If provided, only data points with x in [xrange[0], xrange[1]] are plotted.
    
    Returns
    -------
    (r_value, slope, intercept) : tuple
        r_value : Correlation coefficient of the linear fit.
        slope : Slope of the linear fit.
        intercept : Y-intercept of the linear fit.
    x_fit : np.ndarray
        X values for the fitted line, either spanning the full range of xs or limited to xrange if provided.
    y_fit : np.ndarray
        Corresponding y values of the fitted line at x_fit.
    """
    from scipy.stats import linregress

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
    if xrange:
        x_fit = np.linspace(xrange[0], xrange[1], 200)
        y_fit = slope * x_fit + intercept
    else:
        x_fit = np.linspace(xs.min()*0.9, xs.max()*1.1, 200)
        y_fit = slope * x_fit + intercept
    return (r_value, slope, intercept), x_fit, y_fit


def bondfit(bins, vals, maxfev=2000000):
    """
    Fit a bond-length distribution to a harmonic (Gaussian) potential of the
    form P(x) = a * exp(-k*(x-b)^2 / kBT) to extract a force constant.

    Parameters
    ----------
    bins : array_like
        Bond length values (x-axis of the histogram/distribution).
    vals : array_like
        Probability/density values corresponding to each bin.
    maxfev : int, optional
        Maximum number of function evaluations for curve_fit (default 2000000).

    Returns
    -------
    k : float
        Fitted force constant.
    b : float
        Fitted equilibrium bond length.
    xs : ndarray
        Finely sampled x-values spanning the input bin range, for plotting.
    ys : ndarray
        Fitted curve evaluated at xs.
    """
    from scipy.optimize import curve_fit

    T = 303.0  
    kB = 0.0083144626  # kJ/(mol K)
    kBT = kB * T
    def pb(x, k, a, b):
        return a*np.exp(-k*(x-b)**2/kBT)

    param, cov = curve_fit(pb, bins, vals, p0=[100, 1, 0.0], bounds=([0, 0, 0], [np.inf, np.inf, 1]), maxfev=maxfev)
    k, a, b = param[0], param[1], param[2]
    xs = np.linspace(bins[0], bins[-1], 2000)
    ys = pb(xs, k, a, b)
    return k, b, xs, ys

def anglefit(bins, vals, maxfev=2000000):
    """
    Fit a bond-angle distribution (in degrees) to a harmonic (Gaussian)
    potential of the form P(x) = a * exp(-k*(x-b)^2 / kBT) to extract a
    force constant, working internally in radians.

    Parameters
    ----------
    bins : array_like
        Bond angle values in degrees.
    vals : array_like
        Probability/density values corresponding to each bin.
    maxfev : int, optional
        Maximum number of function evaluations for curve_fit (default 2000000).

    Returns
    -------
    k : float
        Fitted force constant.
    b : float
        Fitted equilibrium angle, in radians.
    xs : ndarray
        Finely sampled x-values (converted back to degrees) spanning the
        input bin range, for plotting.
    ys : ndarray
        Fitted curve evaluated at xs.
    """
    from scipy.optimize import curve_fit
    T = 303.0  
    kB = 0.0083144626  # kJ/(mol K)
    kBT = kB * T
    def pb(x, k, a, b):
        return a*np.exp(-k*(x-b)**2/kBT)
    
    bins = bins/180*np.pi
    param, cov = curve_fit(pb, bins, vals, 
                           p0=[1, 1, 1.5], 
                           bounds=([0, 0, -np.pi], [np.inf, np.inf, np.pi]), 
                           maxfev=maxfev
                           )
    k, a, b = param[0], param[1], param[2]
    xs = np.linspace(bins[0], bins[-1], 2000)
    ys = pb(xs, k, a, b)
    return k, b, xs/np.pi*180, ys

def dihedralfit(bins, vals, multi=0, maxfev=2000000):
    """
    Fit a dihedral-angle distribution (in degrees) to a periodic torsion
    potential of the form P(x) = a * exp(-k*(1+cos(n*x-b)) / kBT), working
    internally in radians, to extract a force constant, phase, and
    periodicity.

    Parameters
    ----------
    bins : array_like
        Dihedral angle values in degrees.
    vals : array_like
        Probability/density values corresponding to each bin.
    maxfev : int, optional
        Maximum number of function evaluations for curve_fit (default 2000000.

    Returns
    -------
    k : float
        Fitted force constant.
    b : float
        Fitted phase offset, in radians.
    n : float
        Fitted periodicity (multiplicity).
    xs : ndarray
        Finely sampled x-values (converted back to degrees) spanning the
        input bin range, for plotting.
    ys : ndarray
        Fitted curve evaluated at xs.
    """
    from scipy.optimize import curve_fit
    T = 303.0  
    kB = 0.0083144626  # kJ/(mol K)
    kBT = kB * T
    def pb(x, k, a, b, n):
        return a*np.exp(-k*(1+np.cos(n*x-b))/kBT)

    bins = bins/180*np.pi

    if multi == 0:
        param, cov = curve_fit(
            pb, bins, vals,
            p0=[10, 0.0001, -0.5, 1],
            bounds=([0, 0, -np.pi, 1], [np.inf, np.inf, np.pi, 5]),
            maxfev=maxfev
        )
        k, a, b, n = param
    else:
        n = multi
        def pb_fixed(x, k, a, b):
            return pb(x, k, a, b, n)

        param, cov = curve_fit(
            pb_fixed, bins, vals,
            p0=[10, 0.0001, -0.5],
            bounds=([0, 0, -np.pi], [np.inf, np.inf, np.pi]),
            maxfev=maxfev
        )
        k, a, b = param

    xs = np.linspace(bins[0], bins[-1], 2000)
    ys = pb(xs, k, a, b, n)
    return k, b, n, xs/np.pi*180, ys

