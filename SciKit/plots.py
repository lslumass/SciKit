import matplotlib as mpl
import matplotlib.pyplot as plt


def general_temp(num_row, num_col, size_x, size_y, *args, **kwargs):
    '''
    create a matplotlib figure;
    plt.subplots(num_row, num_col, figsize=(size_x, size_y))
    '''

    fig, axs = plt.subplots(num_row, num_col, figsize=(size_x, size_y), *args, **kwargs)
    fig.tight_layout(pad=3)
    if num_row*num_col != 1:
        axs = axs.ravel()
        for ax in axs:
            ax.tick_params(axis="both", which='major', direction='in', width=2, length=8.0)
            ax.tick_params(axis="both", which='minor', direction='in', width=2, length=5.0)
            plt.setp(ax.spines.values(), linewidth=2)
    else:
        axs.tick_params(axis="both", which='major', direction='in', width=2, length=8.0)
        ax.tick_params(axis="both", which='minor', direction='in', width=2, length=5.0)
        plt.setp(axs.spines.values(), linewidth=2)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['font.size'] = 18    
    mpl.rcParams['mathtext.default'] = 'regular'

    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['scatter.marker'] = 'o'
    plt.rcParams['lines.markersize'] = 6

    return fig, axs

def set_grid(ax, *args, **kwargs):
    ax.grid(which='major', ls='--', dashes=(5,5), lw=1, alpha=0.5, *args, **kwargs)

def set_legend(ax, *args, **kwargs):
    ax.legend(facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)

## merge the enties with same name and reorder
#  order is the list of entries you expect, like ['line A', 'line B', 'line C'] 
def merge_legend(ax, order=None, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # Remove duplicate labels
    if order is None:
        ax.legend(unique.values(), unique.keys(), facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)  # Set unique legend
    else:
        ordered_handels = [unique[label] for label in order]
        ordered_labels = [label for label in order]
        ax.legend(ordered_handels, ordered_labels, facecolor='white', framealpha=0.7, edgecolor='white', *args, **kwargs)

def set_unique_legend(ax, *args, **kwargs):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), facecolor='white', framealpha=0.7, edgecolor='white')

def number2letter(number, style=1):
    if style ==1 :
        if 1<= number <= 26:
            return chr(number + 64)
        else:
            raise ValueError("Number out of range. Please enter a number between 1 and 26.")
    elif style == 2:
        if 1<= number <= 26:
            return chr(number + 96)
        else:
            raise ValueError("Number out of range. Please enter a number between 1 and 26.")
    else:
        raise ValueError("Style out of range. Please enter a style of 1 or 2.")

def set_label(axs, starting=1, style=1, x=-0.2, y=1.05, **kwargs):
    if style == 1:
        for i, ax in enumerate(axs):
            label = number2letter(i+starting, style=1)
            ax.text(x, y, label, transform=ax.transAxes, size=19, weight='bold', **kwargs)
    elif style == 2:
        for i, ax in enumerate(axs):
            label = '(' + number2letter(i+starting, style=2) + ')'
            ax.text(x, y, label, transform=ax.transAxes, size=19, weight='bold', **kwargs)
    else:
        raise ValueError("Style out of range. Please enter a style of 1 or 2.")
    
def color_cycle(id):
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
    fig.savefig(filename, dpi=600, bbox_inches='tight')


def insert_image(ax, image_path, x, y, zoom=1.0, rotation=0):
    """
    Insert an image into a matplotlib axis.
    
    Parameters:
    - ax: The axis to insert the image into.
    - image_path: Path to the image file.
    - x,y: Coordinates in the axis where the image will be placed.
    - rotate: Angle to rotate the image (default is 0).
    - zoom: Zoom factor for the image (default is 1.0).
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from scipy.ndimage import rotate
    import matplotlib.image as mimg
    
    img = mimg.imread(image_path)
    img = rotate(img, rotation)  # Rotate the image if needed
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)


def add_break(ax, xranges=None, yranges=None):
    """
    xranges: tuple, (x_start, x_end) for x-axis break
    yranges: tuple, (y_start, y_end) for y-axis break
    """
    from break_axes import broken_and_clip_axes
    from break_axes import scale_axes
    if xranges is not None:
        scale_axes(ax, x_interval=[(xranges[0], xranges[1], 0.01)])
        broken_and_clip_axes(ax, x=xranges[0], which='lower')
    if yranges is not None:
        scale_axes(ax, y_interval=[(yranges[0], yranges[1], 0.01)])
        broken_and_clip_axes(ax, y=yranges[0], which='lower')
