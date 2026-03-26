import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def general_temp(num_row, num_col, size_x, size_y, *args, **kwargs):
    """
    Create a standardized matplotlib figure with a grid of subplots.

    Initializes a figure using ``plt.subplots`` and applies consistent
    tick formatting, spine linewidths, font settings, and default line/marker
    styles across all axes. Intended as a base template for publication-ready figures.

    Parameters
    ----------
    num_row : int
        Number of subplot rows.
    num_col : int
        Number of subplot columns.
    size_x : float
        Figure width in inches.
    size_y : float
        Figure height in inches.
    *args
        Additional positional arguments forwarded to ``plt.subplots``.
    **kwargs
        Additional keyword arguments forwarded to ``plt.subplots``
        (e.g., ``sharex``, ``sharey``, ``subplot_kw``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axs : matplotlib.axes.Axes or numpy.ndarray of Axes
        A single ``Axes`` object if ``num_row * num_col == 1``,
        otherwise a flattened 1-D array of ``Axes`` objects.

    Notes
    -----
    - Major ticks: inward, width=2, length=8.
    - Minor ticks: inward, width=1.5, length=4.
    - Spine linewidth: 2.
    - Font: Arial, size 18.
    - Default line width: 2.0; marker: 'o', size 6.

    Examples
    --------
    >>> fig, axs = general_temp(2, 3, 12, 8)
    >>> axs[0].plot([1, 2, 3], [4, 5, 6])
    >>> fig.savefig("output.png")
    """
    fig, axs = plt.subplots(num_row, num_col, figsize=(size_x, size_y), *args, **kwargs)
    fig.tight_layout(pad=3)
    if num_row*num_col != 1:
        axs = axs.ravel()
        for ax in axs:
            ax.tick_params(axis="both", which='major', direction='in', width=2, length=8.0)
            ax.tick_params(axis="both", which='minor', direction='in', width=1.5, length=4.0)
            plt.setp(ax.spines.values(), linewidth=2)
    else:
        axs.tick_params(axis="both", which='major', direction='in', width=2, length=8.0)
        axs.tick_params(axis="both", which='minor', direction='in', width=1.5, length=4.0)
        plt.setp(axs.spines.values(), linewidth=2)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['font.size'] = 18    
    mpl.rcParams['mathtext.default'] = 'regular'

    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['scatter.marker'] = 'o'
    plt.rcParams['lines.markersize'] = 6

    return fig, axs


def set_grid(ax, *args, **kwargs):
    """
    Add a styled major grid to an axis.

    Draws dashed major gridlines with reduced opacity, suitable for
    background reference without overwhelming plotted data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes on which the grid will be drawn.
    *args
        Additional positional arguments forwarded to ``ax.grid``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.grid``
        (e.g., ``color``, ``axis``).

    Notes
    -----
    Grid style defaults: dashes=(5,5), linewidth=1, alpha=0.5.

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 6, 4)
    >>> set_grid(axs)
    """
    ax.grid(which='major', ls='--', dashes=(5,5), lw=1, alpha=0.5, *args, **kwargs)


def set_legend(ax, *args, **kwargs):
    """
    Add a styled legend to an axis.

    Applies a white semi-transparent background with no visible edge,
    keeping the legend readable without obstructing the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes on which the legend will be placed.
    *args
        Additional positional arguments forwarded to ``ax.legend``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.legend``
        (e.g., ``loc``, ``ncol``, ``fontsize``).

    Examples
    --------
    >>> ax.plot([1, 2], [3, 4], label='Data A')
    >>> set_legend(ax, loc='upper left')
    """
    ax.legend(facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)


def merge_legend(ax, order=None, *args, **kwargs):
    """
    Deduplicate legend entries and optionally reorder them.

    Useful when multiple plot calls share the same label (e.g., in a loop)
    and only one representative legend entry is desired per label.
    An explicit ``order`` list controls the final display sequence.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes whose legend will be rebuilt.
    order : list of str, optional
        Ordered list of label strings defining the desired legend sequence.
        All labels in ``order`` must already exist among the plotted artists.
        If ``None``, unique labels are displayed in their first-occurrence order.
    *args
        Additional positional arguments forwarded to ``ax.legend``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.legend``
        (e.g., ``loc``, ``ncol``).

    Raises
    ------
    KeyError
        If a label in ``order`` does not match any plotted artist label.

    Examples
    --------
    >>> for val in data:
    ...     ax.plot(x, val, color='blue', label='Series A')
    >>> merge_legend(ax, order=['Series A', 'Series B'])
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # Remove duplicate labels
    if order is None:
        ax.legend(unique.values(), unique.keys(), facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)
    else:
        ordered_handels = [unique[label] for label in order]
        ordered_labels = [label for label in order]
        ax.legend(ordered_handels, ordered_labels, facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)


def set_unique_legend(ax, *args, **kwargs):
    """
    Add a legend showing only the first occurrence of each unique label.

    Operates on the current active axes (``plt.gca()``) to collect handles
    and labels, deduplicates them by label string, then renders the legend
    on the provided ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes on which the deduplicated legend will be placed.
    *args
        Additional positional arguments (currently unused; reserved for
        future forwarding to ``ax.legend``).
    **kwargs
        Additional keyword arguments (currently unused; reserved for
        future forwarding to ``ax.legend``).

    Notes
    -----
    This function reads handles from ``plt.gca()``, not from ``ax`` directly.
    For multi-axes figures, prefer :func:`merge_legend` to avoid ambiguity.

    Examples
    --------
    >>> ax.plot(x, y1, label='Experiment')
    >>> ax.plot(x, y2, label='Experiment')   # duplicate label
    >>> set_unique_legend(ax)
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), facecolor='white', framealpha=0.7, edgecolor='white')


def number2letter(number, style=1):
    """
    Convert an integer (1–26) to its corresponding alphabetic character.

    Parameters
    ----------
    number : int
        Integer in the range [1, 26] to convert.
    style : {1, 2}, optional
        Output case style:

        - ``1`` (default): uppercase letter (e.g., 1 → 'A').
        - ``2``: lowercase letter (e.g., 1 → 'a').

    Returns
    -------
    str
        A single uppercase or lowercase letter.

    Raises
    ------
    ValueError
        If ``number`` is outside [1, 26] or ``style`` is not 1 or 2.

    Examples
    --------
    >>> number2letter(3)
    'C'
    >>> number2letter(3, style=2)
    'c'
    """
    if style == 1:
        if 1 <= number <= 26:
            return chr(number + 64)
        else:
            raise ValueError("Number out of range. Please enter a number between 1 and 26.")
    elif style == 2:
        if 1 <= number <= 26:
            return chr(number + 96)
        else:
            raise ValueError("Number out of range. Please enter a number between 1 and 26.")
    else:
        raise ValueError("Style out of range. Please enter a style of 1 or 2.")


def set_label(axs, starting=1, style=1, x=-0.2, y=1.05, **kwargs):
    """
    Annotate a sequence of axes with alphabetic panel labels.

    Places a bold letter label (e.g., 'A', 'B', … or '(a)', '(b)', …)
    in the upper-left region of each axis using axis-relative coordinates.
    Commonly used for multi-panel publication figures.

    Parameters
    ----------
    axs : iterable of matplotlib.axes.Axes
        Sequence of axes to label, processed in order.
    starting : int, optional
        Integer index of the first label (default ``1`` → 'A' or '(a)').
    style : {1, 2}, optional
        Label format:

        - ``1`` (default): uppercase letter only (e.g., 'A', 'B', 'C').
        - ``2``: lowercase letter in parentheses (e.g., '(a)', '(b)', '(c)').
    x : float, optional
        Horizontal position in axis-relative coordinates (default ``-0.2``).
    y : float, optional
        Vertical position in axis-relative coordinates (default ``1.05``).
    **kwargs
        Additional keyword arguments forwarded to ``ax.text``
        (e.g., ``color``, ``fontsize``).

    Raises
    ------
    ValueError
        If ``style`` is not 1 or 2.

    Examples
    --------
    >>> fig, axs = general_temp(1, 3, 12, 4)
    >>> set_label(axs)               # labels: A, B, C
    >>> set_label(axs, style=2)      # labels: (a), (b), (c)
    >>> set_label(axs, starting=4)   # labels: D, E, F
    """
    if style == 1:
        for i, ax in enumerate(axs):
            label = number2letter(i + starting, style=1)
            ax.text(x, y, label, transform=ax.transAxes, size=19, weight='bold', **kwargs)
    elif style == 2:
        for i, ax in enumerate(axs):
            label = '(' + number2letter(i + starting, style=2) + ')'
            ax.text(x, y, label, transform=ax.transAxes, size=19, weight='bold', **kwargs)
    else:
        raise ValueError("Style out of range. Please enter a style of 1 or 2.")


def color_cycle(id):
    """
    Return a hex color string from a fixed 10-color palette.

    Mirrors matplotlib's default ``tab10`` color cycle, providing
    convenient index-based access for consistent multi-series coloring.

    Parameters
    ----------
    id : int
        Color index in the range [1, 10]:

        ====  =========  ===================
        id    hex        approximate color
        ====  =========  ===================
        1     #1f77b4    muted blue
        2     #ff7f0e    safety orange
        3     #2ca02c    cooked asparagus green
        4     #d62728    brick red
        5     #9467bd    muted purple
        6     #8c564b    chestnut brown
        7     #e377c2    raspberry pink
        8     #7f7f7f    middle gray
        9     #bcbd22    curry yellow-green
        10    #17becf    blue-teal
        ====  =========  ===================

    Returns
    -------
    str
        Hex color string. Returns ``'#000000'`` (black) for any ``id``
        not in [1, 10].

    Examples
    --------
    >>> ax.plot(x, y, color=color_cycle(1))   # muted blue
    >>> ax.plot(x, z, color=color_cycle(2))   # safety orange
    """
    colors = {
        1 : '#1f77b4',
        2 : '#ff7f0e',
        3 : '#2ca02c',
        4 : '#d62728',
        5 : '#9467bd',
        6 : '#8c564b',
        7 : '#e377c2',
        8 : '#7f7f7f',
        9 : '#bcbd22',
        10: '#17becf',
    }
    return colors.get(id, '#000000')


def savefig(fig, filename):
    """
    Save a figure to disk at publication-quality resolution.

    Wraps ``fig.savefig`` with ``dpi=600`` and ``bbox_inches='tight'``
    to ensure all artists (titles, labels, legends) are included without
    clipping and the output is suitable for journal submission.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str or path-like
        Output file path, including extension (e.g., ``'fig1.png'``,
        ``'fig1.pdf'``, ``'fig1.svg'``). The format is inferred from
        the extension.

    Examples
    --------
    >>> fig, axs = general_temp(1, 2, 10, 4)
    >>> savefig(fig, 'results/figure1.png')
    """
    fig.savefig(filename, dpi=600, bbox_inches='tight')


def insert_image(ax, image_path, x, y, zoom=1.0, rotation=0):
    """
    Embed a raster image into a matplotlib axis at a specified data coordinate.

    Reads an image file, optionally rotates it, and places it as an
    ``AnnotationBbox`` artist anchored to the given data-space coordinates.
    Useful for inset schematics, icons, or experimental photographs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes into which the image will be inserted.
    image_path : str or path-like
        Path to the image file (any format supported by
        ``matplotlib.image.imread``, e.g., PNG, JPEG).
    x : float
        Horizontal anchor position in data coordinates.
    y : float
        Vertical anchor position in data coordinates.
    zoom : float, optional
        Scaling factor applied to the image (default ``1.0``; values
        below 1 shrink, above 1 enlarge).
    rotation : float, optional
        Counter-clockwise rotation angle in degrees (default ``0``).
        Uses ``scipy.ndimage.rotate`` with nearest-neighbor interpolation.

    Notes
    -----
    Rotation via ``scipy.ndimage.rotate`` may introduce black borders around
    the image corners. For transparent PNGs, consider passing
    ``reshape=False`` by patching the call if border artifacts occur.

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 6, 6)
    >>> insert_image(axs, 'schematic.png', x=0.5, y=0.5, zoom=0.3, rotation=45)
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from scipy.ndimage import rotate
    import matplotlib.image as mimg

    img = mimg.imread(image_path)
    img = rotate(img, rotation)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)


def add_break(ax, xranges=None, yranges=None):
    """
    Insert an axis break to omit a continuous range of x- or y-values.

    Compresses the specified interval to near-zero width/height using
    ``break_axes.scale_axes``, then adds a visual break indicator via
    ``break_axes.broken_and_clip_axes``. Helpful when data has a large
    gap that would otherwise compress the regions of interest.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes on which the break will be applied.
    xranges : tuple of float, optional
        ``(x_start, x_end)`` defining the x-axis interval to collapse.
        If ``None``, no x-break is applied.
    yranges : tuple of float, optional
        ``(y_start, y_end)`` defining the y-axis interval to collapse.
        If ``None``, no y-break is applied.

    Notes
    -----
    Requires the third-party ``break_axes`` package. Both ``xranges`` and
    ``yranges`` can be provided simultaneously to apply breaks on both axes.
    The break marker is placed at the lower boundary of each specified range.

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 6, 4)
    >>> axs.plot([0, 1, 10, 11], [0, 1, 2, 3])
    >>> add_break(axs, xranges=(2, 9))   # hide x ∈ [2, 9]
    """
    from break_axes import broken_and_clip_axes
    from break_axes import scale_axes
    if xranges is not None:
        scale_axes(ax, x_interval=[(xranges[0], xranges[1], 0.01)])
        broken_and_clip_axes(ax, x=xranges[0], which='lower')
    if yranges is not None:
        scale_axes(ax, y_interval=[(yranges[0], yranges[1], 0.01)])
        broken_and_clip_axes(ax, y=yranges[0], which='lower')


def plot_contacts(ax, contacts, cmap, x_shift, y_shift, **kwargs):
    """
    Display a 2-D contact matrix as a raster image on an axis.

    Wraps ``ax.imshow`` to render a square or rectangular NumPy array
    with its origin at the bottom-left, offset by the specified shifts.
    Designed for contact maps (e.g., Hi-C, distance matrices) where
    axis coordinates must align with external data indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes on which the contact matrix will be displayed.
    contacts : numpy.ndarray, shape (Nx, Ny)
        2-D array of contact values. ``Nx`` sets the x-extent and
        ``Ny`` sets the y-extent of the displayed image.
    cmap : str or matplotlib.colors.Colormap
        Colormap applied to the contact values
        (e.g., ``'hot_r'``, ``'RdBu_r'``).
    x_shift : float
        Offset applied to the left and right edges of the image extent,
        shifting the entire matrix along the x-axis.
    y_shift : float
        Offset applied to the bottom and top edges of the image extent,
        shifting the entire matrix along the y-axis.
    **kwargs
        Additional keyword arguments forwarded to ``ax.imshow``
        (e.g., ``vmin``, ``vmax``, ``norm``, ``interpolation``).

    Returns
    -------
    im : matplotlib.image.AxesImage
        The ``AxesImage`` object returned by ``ax.imshow``, which can
        be passed to ``plt.colorbar`` for a color scale.

    Examples
    --------
    >>> matrix = np.random.rand(100, 100)
    >>> fig, axs = general_temp(1, 1, 6, 5)
    >>> im = plot_contacts(axs, matrix, cmap='hot_r', x_shift=0, y_shift=0)
    >>> fig.colorbar(im, ax=axs, label='Contact frequency')
    """
    Nx = contacts.shape[0]
    Ny = contacts.shape[1]
    im = ax.imshow(contacts, origin='lower', extent=[x_shift, Nx+x_shift, y_shift, Ny+y_shift], cmap=cmap, **kwargs)
    return im


import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_hist2d(ax, x, y, bins=50, log_scale=False, contours=False, **kwargs):
    """
    Create a 2D histogram with optional contours.

    Parameters:
    -----------
    x : array-like, shape (n_samples, n_features) or (n_samples,)
        X data
    y : array-like, shape (n_samples, n_features) or (n_samples,)
        Y data
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    bins : int or array-like, default=50
        Number of bins for histogram
    log_scale : bool, default=False
        Whether to use log scale for colors
    contours : bool, default=False
        Whether to overlay contour lines
    **kwargs : dict
        Additional customization options:
          colorbar           : bool,                    default=False
          cbar_size          : str,                     default='3%'
          cbar_pad           : float,                   default=0.1
          cmap               : str or LinearSegmentedColormap, default='viridis'
          alpha              : float,                   default=1.0
          n_contours         : int,                     default=5
          contour_colors     : str,                     default='black'
          contour_alpha      : float,                   default=0.6
          contour_linewidths : float,                   default=1.0
          contour_labels     : bool,                    default=False
          override_contour_levels : array-like,         optional
          grid               : bool,                    default=True

    Returns:
    --------
    hist : 2D histogram array
    xedges, yedges : bin edges
    cbar_ax : colorbar axes object (None if colorbar=False)
    """

    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Flatten arrays if they are 2D
    x_flat = x.flatten() if x.ndim > 1 else x.copy()
    y_flat = y.flatten() if y.ndim > 1 else y.copy()

    # Remove any NaN values
    mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
    x_clean = x_flat[mask]
    y_clean = y_flat[mask]

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=[bins, bins], density=True)

    # Handle log scale
    hist_plot = hist.copy()
    hist_plot[hist_plot == 0] = np.nan  # White for empty bins

    norm = None
    if log_scale:
        min_val = np.nanmin(hist_plot[hist_plot > 0]) if np.any(hist_plot > 0) else 1
        max_val = np.nanmax(hist_plot)
        if max_val > min_val:
            norm = LogNorm(vmin=min_val, vmax=max_val)
        else:
            log_scale = False

    # Set colormap with white for NaN/empty bins
    cmap = kwargs.get('cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap].copy()
    cmap.set_bad(kwargs.get('bad_color', 'none'))

    # Plot the 2D histogram
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, hist_plot.T,
                       cmap=cmap,
                       norm=norm,
                       shading='flat',
                       alpha=kwargs.get('alpha', 1.0))

    # Add contours if requested
    if contours and np.any(hist > 0):
        n_contours = kwargs.get('n_contours', 5)

        if kwargs.get('override_contour_levels') is not None:
            contour_levels = kwargs['override_contour_levels']
        elif log_scale and norm is not None:
            contour_levels = np.logspace(np.log10(min_val), np.log10(max_val), n_contours)
        else:
            contour_levels = np.linspace(np.nanmin(hist_plot), np.nanmax(hist_plot), n_contours)

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X_contour, Y_contour = np.meshgrid(x_centers, y_centers)

        cs = ax.contour(X_contour, Y_contour, hist_plot.T,
                        levels=contour_levels,
                        colors=kwargs.get('contour_colors', 'black'),
                        alpha=kwargs.get('contour_alpha', 0.6),
                        linewidths=kwargs.get('contour_linewidths', 1.0))

        if kwargs.get('contour_labels', False):
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Grid
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)

    # Optional colorbar
    cbar_ax = None
    if kwargs.get('colorbar', False):
        try:
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right",
                                          size=kwargs.get('cbar_size', '3%'),
                                          pad=kwargs.get('cbar_pad', 0.1))
            ax.get_figure().colorbar(im, cax=cbar_ax)
        except Exception:
            pass

    return hist, xedges, yedges, cbar_ax

def plot_hist2d_contour(ax, x, y, levels=5, fill_color="steelblue", fill_alpha=0.4,
                        line_color="steelblue", line_alpha=0.9, linewidths=1.5,
                        gridsize=100, bw_method="scott",
):
    """
    Plot a 2D histogram as filled contours with contour lines on a given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array-like of shape (N,)
    levels : int, float, or array-like
        - int   → number of auto-spaced levels.
        - float → single density threshold; fills the region above it.
        - array → explicit iso-density boundaries (must have >= 2 values).
    fill_color : str or RGB tuple
    fill_alpha : float
    line_color : str or RGB tuple
    line_alpha : float
    linewidths : float or sequence of float
    gridsize : int
    bw_method : str, scalar, or callable

    Returns
    -------
    cf : QuadContourSet  (filled)
    cl : QuadContourSet  (lines)
    """
    import numpy as np
    from scipy.stats import gaussian_kde

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Grid with a small padding so the outermost contour closes cleanly
    pad_x = (x.max() - x.min()) * 0.05
    pad_y = (y.max() - y.min()) * 0.05
    xi = np.linspace(x.min() - pad_x, x.max() + pad_x, gridsize)
    yi = np.linspace(y.min() - pad_y, y.max() + pad_y, gridsize)
    Xi, Yi = np.meshgrid(xi, yi)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # ── Normalise `levels` so contourf always receives >= 2 boundaries ────────
    if np.ndim(levels) == 0:                  # scalar int or float
        levels = int(levels) if float(levels) == int(levels) else float(levels)
        if isinstance(levels, int):            # e.g. levels=5  → auto spacing
            pass                               # let matplotlib handle it
        else:                                  # e.g. levels=0.0032 → threshold
            levels = [levels, float(Zi.max())]
    # array-like: pass through unchanged (user's responsibility to have >= 2)
    # ──────────────────────────────────────────────────────────────────────────

    cf = ax.contourf(
        Xi, Yi, Zi,
        levels=levels,
        colors=[fill_color],
        alpha=fill_alpha,
    )
    cl = ax.contour(
        Xi, Yi, Zi,
        levels=cf.levels,
        colors=[line_color],
        alpha=line_alpha,
        linewidths=linewidths,
    )
    return cf, cl


class DualYAxis:
    """
    A wrapper around a matplotlib Axes that supports independent y-axis coloring
    for dual-axis (twinx) figures.

    Do not instantiate directly. Use :func:`dualY` to create a pair.

    All standard ``Axes`` methods (``plot``, ``set_ylabel``, ``set_xlim``, etc.)
    are transparently forwarded to the underlying axes via ``__getattr__``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The underlying axes to wrap.
    side : {'left', 'right'}
        Which y-axis spine this object owns. ``'left'`` for the original axes,
        ``'right'`` for the twinx axes.

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 8, 5)
    >>> ax1, ax2 = dualY(axs)
    >>> ax1.plot(x, y1, color=color_cycle(1))
    >>> ax2.plot(x, y2, color=color_cycle(2))
    >>> ax1.set_ylabel('Temperature (°C)')
    >>> ax2.set_ylabel('Pressure (Pa)')
    >>> ax1.set_color(color_cycle(1))
    >>> ax2.set_color(color_cycle(2))
    """

    def __init__(self, ax, side):
        object.__setattr__(self, '_ax', ax)
        object.__setattr__(self, '_side', side)

    def set_color(self, color):
        """
        Apply a uniform color to this axis's y-spine, ticks, tick labels, and label.

        Parameters
        ----------
        color : color-like
            Any matplotlib-compatible color string or tuple
            (e.g. ``'#1f77b4'``, ``'red'``, ``(0.1, 0.5, 0.9)``).

        Examples
        --------
        >>> ax1.set_color('#1f77b4')
        >>> ax2.set_color('#ff7f0e')
        """
        ax   = object.__getattribute__(self, '_ax')
        side = object.__getattribute__(self, '_side')

        ax.spines[side].set_edgecolor(color)
        ax.tick_params(axis='y', colors=color)
        ax.yaxis.label.set_color(color)

    def __getattr__(self, name):
        ax = object.__getattribute__(self, '_ax')
        return getattr(ax, name)

    def __setattr__(self, name, value):
        ax = object.__getattribute__(self, '_ax')
        setattr(ax, name, value)


def dualY(ax):
    """
    Set up a dual y-axis on an existing axes and return two :class:`DualYAxis`
    wrappers — one for each y-axis — whose :meth:`~DualYAxis.set_color` method
    colors the spine, ticks, tick labels, and axis label together.

    The left spine belongs to ``ax1``; the right spine belongs to ``ax2``.
    Redundant inner spines are hidden so the frame stays clean.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A single axes object, typically one element from the array returned by
        :func:`general_temp`.

    Returns
    -------
    ax1 : DualYAxis
        Wraps the original ``ax``. Owns the **left** y-axis.
    ax2 : DualYAxis
        Wraps a new ``ax.twinx()``. Owns the **right** y-axis.

    Notes
    -----
    - ``ax1`` and ``ax2`` proxy all standard ``Axes`` calls, so you can use
      ``ax1.plot(...)``, ``ax1.set_xlim(...)``, ``ax2.set_ylabel(...)``, etc.
      as normal.
    - Call :meth:`~DualYAxis.set_color` *after* setting ``ylabel`` so the
      label color is applied correctly.
    - The x-axis ticks and spine color are not modified; style them via
      ``ax1.tick_params(axis='x', ...)`` as usual.

    Examples
    --------
    >>> fig, axs = general_temp(1, 1, 8, 5)
    >>> ax1, ax2 = dualY(axs)
    >>>
    >>> ax1.plot(x, temp,     color=color_cycle(1), label='Temperature')
    >>> ax2.plot(x, pressure, color=color_cycle(2), label='Pressure')
    >>>
    >>> ax1.set_xlabel('Time (s)')
    >>> ax1.set_ylabel('Temperature (°C)')
    >>> ax2.set_ylabel('Pressure (Pa)')
    >>>
    >>> ax1.set_color(color_cycle(1))
    >>> ax2.set_color(color_cycle(2))
    """
    twin = ax.twinx()

    ax.spines['right'].set_visible(False)
    twin.spines['left'].set_visible(False)

    twin.tick_params(axis='both', which='major', direction='in', width=2, length=8.0)
    twin.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4.0)
    plt.setp(twin.spines.values(), linewidth=2)

    ax1 = DualYAxis(ax,   side='left')
    ax2 = DualYAxis(twin, side='right')
    return ax1, ax2