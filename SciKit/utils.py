import numpy as np
import MDAnalysis as mda


def scatter2hist(point_list, num_bin, styles):
    '''
    Convert scatter points to histogram or distribution.
    styles:
        pdf   : histogram of probability density function
        pmf   : histogram of probability mass function
        s_pdf : smoothed line of pdf
        s_pmf : smoothed line of pmf
    '''
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
    '''
    calculate the pc1, pc2 from the pca analysis of psf and dcd file
    '''
    from MDAnalysis.analysis import pca

    u = mda.Universe(psf, dcd)
    atomgroup = u.select_atoms(sel)
    pc = pca.PCA(u, select=sel, align=align).run()
    pc1, pc2 = pc.transform(atomgroup)[:,0], pc.transform(atomgroup)[:,1]
    var1, var2 = pc.cumulated_variance[0], pc.cumulated_variance[1]-pc.cumulated_variance[0]
    return pc1, pc2, var1, var2


def pca2fe(pc1, pc2, num_bin=100):
    '''
    convert pc1, pc2 to free energy distribution in the unit of kBT
    '''
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
    im = axs.imshow(fe.T, origin='lower', cmap=cmap)
    axs.figure.colorbar(im, ax = axs, label='Free energy (k$_B$T)', fraction=0.046)
    if var1 == None:
        axs.set(xlabel='PC1', xticks=[], ylabel='PC2', yticks=[])
    else:
        axs.set(xlabel=f'PC1 ({var1*100:.2f}%)', xticks=[], ylabel=f'PC2 ({var2*100:.2f}%)', yticks=[])
    
    if contour:
        axs.contour(fe.T, levels=levels, colors='k')

def pca2d(axs, psf, dcd, sel, num_bin=100, align=True, cmap='viridis'): 
    from MDAnalysis.analysis import pca

    u = mda.Universe(psf, dcd)
    atomgroup = u.select_atoms(sel)
    pc = pca.PCA(u, select=sel, align=align).run()
    pc1, pc2 = pc.transform(atomgroup)[:,0], pc.transform(atomgroup)[:,1]
    var1, var2 = pc.cumulated_variance[0], pc.cumulated_variance[1]-pc.cumulated_variance[0]
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=num_bin)
    pmf = H / np.sum(H)

    # convert pmf to free energy
    fe = -np.log(pmf)
    fe_min = np.min(fe[np.isfinite(fe)])
    fe -= fe_min

    im = axs.imshow(fe.T, origin='lower', cmap=cmap)
    axs.figure.colorbar(im, ax = axs, label='Free energy (k$_B$T)', fraction=0.046)
    axs.set(xlabel=f'PC1 ({var1*100:.2f}%)', xticks=[], ylabel=f'PC2 ({var2*100:.2f}%)', yticks=[])
    


def block_mean(data, division):
    '''
    calculate the block mean and std of a list of number 
    data: a list of the scatter points along simulation time
    division: how many parts should the data divided, and the first part will be ignored for analysis
                the rest part will be divided into two blocks, and then the standard deviation and average
    '''
    l = len(data)
    start_frame = int(l/division)
    half = start_frame + int((l-start_frame)/2)
    part1 = np.mean(data[start_frame:half])
    part2 = np.mean(data[half:])
    average = np.mean([part1, part2])
    error = np.std([part1, part2])
    return average, error

def rms(data):
    data = np.asarray(data)
    return np.sqrt(np.mean(data**2))

def block_bootstrap(data, blocksize=1000, nsamples=1000, statistic=rms):
    '''
    calculate the block bootstrap standard deviation
    Pramater:
        data: 1D array of data points
        blocksize: int, size of each block, default=100
        nsamples: number of bootstrap samples, default=1000
        statistic: function to calculate the statistic of interest, like np.mean, default is rms
    Return:
        Block bootstrap standard deviation
    '''
    data = np.asarray(data)
    n = len(data)
    bootstrap_stats = []
    for _ in range(nsamples):
        # Generate bootstrap sample using circular block bootstrap
        bootstrap_sample = []
        while len(bootstrap_sample) < n:
            # Randomly select a starting index
            start_idx = np.random.randint(0, n)
            # Extract a block (with circular wrapping)
            block = []
            for i in range(blocksize):
                block.append(data[(start_idx + i) % n])
            bootstrap_sample.extend(block)
        
        # trim to exact length
        bootstrap_sample = bootstrap_sample[:n]
        # calculate statistic
        bootstrap_stats.append(statistic(bootstrap_sample))

    # calculate standard deviation
    sd = np.std(bootstrap_stats, ddof=1)

    return sd


def remove_zero(xs, ys):
    '''
    remove the zero section at the beginning and ending of one distribution
    xs: data for x-axis
    ys: data for y-axis
    '''
    for idx in range(len(xs)):
        # find the first non-zero point from begging
        if ys[idx] != 0:
            start = idx-5
            break
    for idx in reversed(range(len(xs))):
        # find the last non-zero point at the end
        if ys[idx] != 0:
            end = idx+5
            break
    return(xs[start], xs[end])


def myload(filename, *args, **kwargs):
    data = np.loadtxt(filename, *args, **kwargs)
    cols = len(data[0, :])
    results = []
    for col in range(cols):
        results.append(data[:, col])
    results = results + [0, 0, 0, 0, 0, 0, 0, 0]
    return tuple(results)
