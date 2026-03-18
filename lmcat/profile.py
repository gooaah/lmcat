"""
lmcat.profile — histogram construction and Gaussian-smoothed density profiles.
"""
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def trajectory_histogram(traj, hist_range, bins, slab_sym):
    """
    Build a z-density histogram for one atomic species across a trajectory.

    The histogram is area-normalized so the result is in units of
    atoms/Angstrom^3 (per bin width).

    Parameters
    ----------
    traj : list of ase.Atoms
        MD trajectory (list of snapshots).
    hist_range : tuple of float
        (z_min, z_max) histogram range [Angstrom].
    bins : int
        Number of histogram bins.
    slab_sym : str
        Chemical symbol of the species to histogram, e.g. ``'Cu'``.

    Returns
    -------
    zs : np.ndarray
        Bin-centre z-coordinates, shape (bins,) [Angstrom].
    slab_hist : np.ndarray
        Averaged, area-normalized histogram, shape (bins,)
        [atoms/Angstrom^3 per bin width].

    Notes
    -----
    The area normalization uses the xy-plane cross-section of the first
    snapshot's cell.  This assumes an orthogonal simulation cell.
    """
    cell = traj[0].get_cell()
    area = cell[0, 0] * cell[1, 1]

    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    slab_indice = [i for i, s in enumerate(symbols) if s == slab_sym]

    hist = np.zeros(bins)
    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:, 2]
        h, bin_edges = np.histogram(zpos[slab_indice], bins=bins,
                                    range=hist_range, density=False)
        hist += h

    binsize = (hist_range[1] - hist_range[0]) / bins
    slab_hist = hist / len(traj) / area / binsize
    zs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return zs, slab_hist


def trajectory_histogram_full(traj, hist_range, bins, num_surf,
                              slab_sym='Cu', layer_syms=None):
    """
    Build z-density histograms for the slab and an adsorbed layer, and
    compute structural statistics.

    Unlike :func:`trajectory_histogram`, the returned histograms are
    **not** area-normalized; the normalization is deferred to
    :func:`smooth_density_two_species`.

    Parameters
    ----------
    traj : list of ase.Atoms
        MD trajectory.
    hist_range : tuple of float
        (z_min, z_max) histogram range [Angstrom].
    bins : int
        Number of histogram bins.
    num_surf : int
        Number of topmost slab atoms used to estimate the surface position.
    slab_sym : str, optional
        Chemical symbol of the slab species. Default: ``'Cu'``.
    layer_syms : list of str, optional
        Chemical symbols of the adsorbed-layer species. Default: ``['C']``.

    Returns
    -------
    c_hist : np.ndarray
        Raw (un-normalized) layer histogram, shape (bins,).
    slab_hist : np.ndarray
        Raw (un-normalized) slab histogram, shape (bins,).
    c_mean : float
        Time-averaged mean z-position of the layer atoms [Angstrom].
    s_dist : float
        Time-averaged distance between the layer mean z and the topmost
        *num_surf* slab atoms [Angstrom].
    area : float
        xy cross-sectional area of the simulation cell [Angstrom^2].

    Notes
    -----
    Call :func:`smooth_density_two_species` to convert the raw histograms
    to physical density profiles in atoms/Angstrom^3.
    """
    if layer_syms is None:
        layer_syms = ['C']

    cell = traj[0].get_cell()
    area = cell[0, 0] * cell[1, 1]

    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    c_indice = [i for i, s in enumerate(symbols) if s in layer_syms]
    slab_indice = [i for i, s in enumerate(symbols) if s == slab_sym]

    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)
    s_dist = 0.0
    c_mean = 0.0
    slab_max = 0.0
    c_z_pos = np.array([])
    c_z_std = 0.0
    c_thick = 0.0

    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:, 2]
        z1 = zpos[c_indice]
        z2 = zpos[slab_indice]
        c_z_pos = np.concatenate([c_z_pos, z1])
        c_z_std += np.std(z1)
        c_thick += z1.max() - z1.min()
        h1, _ = np.histogram(z1, bins=bins, range=hist_range)
        hist1 += h1
        h2, _ = np.histogram(z2, bins=bins, range=hist_range)
        hist2 += h2
        c_mean += z1.mean()
        slab_max += z2.max()
        surf_zs = z2[np.argsort(z2)[-num_surf:]]
        s_dist += z1.mean() - surf_zs.mean()

    n = len(traj)
    c_mean /= n
    slab_max /= n
    c_z_std /= n
    c_thick /= n
    s_dist /= n

    logger.info("Average layer height: %.4f Angstrom", c_mean)
    logger.info("Layer z std (all snapshots): %.4f Angstrom", np.std(c_z_pos))
    logger.info("Layer z std (avg over snapshots): %.4f Angstrom", c_z_std)
    logger.info("Layer thickness (max-min, avg): %.4f Angstrom", c_thick)
    logger.info("layer_mean - slab_max: %.4f Angstrom", c_mean - slab_max)
    logger.info("Solid dist (layer mean - top slab atoms): %.4f Angstrom", s_dist)

    c_hist = hist1 / n
    slab_hist = hist2 / n

    return c_hist, slab_hist, c_mean, s_dist, area


def smooth_density(slab_hist, gauss_width, w_bin):
    """
    Convert an area-normalized histogram to a Gaussian-smoothed density profile.

    Parameters
    ----------
    slab_hist : np.ndarray
        Area-normalized histogram [atoms/Angstrom^3 per bin width], as
        returned by :func:`trajectory_histogram`.
    gauss_width : float
        Standard deviation of the Gaussian smoothing kernel [Angstrom].
    w_bin : float
        Bin width [Angstrom].

    Returns
    -------
    slab_den : np.ndarray
        Smoothed density profile [atoms/Angstrom^3], same shape as
        *slab_hist*.

    Notes
    -----
    The input *slab_hist* must already be area-normalized (as produced by
    :func:`trajectory_histogram`).  If using the raw histograms from
    :func:`trajectory_histogram_full`, use :func:`smooth_density_two_species`
    instead.
    """
    slab_sigma = gauss_width / w_bin
    logger.debug("Bin width: %.4f Angstrom", w_bin)
    logger.debug("Gaussian width for slab: %.4f Angstrom (%.2f bins)", gauss_width, slab_sigma)
    slab_den = gaussian_filter1d(slab_hist, slab_sigma) / w_bin
    return slab_den


def smooth_density_two_species(slab_hist, c_hist, gauss_width_slab,
                               gauss_width_C, w_bin, area):
    """
    Convert raw (un-normalized) histograms for two species to Gaussian-smoothed
    density profiles.

    Parameters
    ----------
    slab_hist : np.ndarray
        Raw slab histogram (counts per bin, averaged over frames), as
        returned by :func:`trajectory_histogram_full`.
    c_hist : np.ndarray
        Raw layer histogram (counts per bin, averaged over frames).
    gauss_width_slab : float
        Gaussian smoothing width for the slab species [Angstrom].
    gauss_width_C : float
        Gaussian smoothing width for the layer species [Angstrom].
    w_bin : float
        Bin width [Angstrom].
    area : float
        xy cross-sectional area [Angstrom^2].

    Returns
    -------
    slab_den : np.ndarray
        Smoothed slab density profile [atoms/Angstrom^3].
    c_den : np.ndarray
        Smoothed layer density profile [atoms/Angstrom^3].
    """
    slab_sigma = gauss_width_slab / w_bin
    c_sigma = gauss_width_C / w_bin
    logger.debug("Bin width: %.4f Angstrom", w_bin)
    logger.debug("Gaussian width for slab: %.4f Angstrom (%.2f bins)", gauss_width_slab, slab_sigma)
    logger.debug("Gaussian width for layer: %.4f Angstrom (%.2f bins)", gauss_width_C, c_sigma)
    slab_den = gaussian_filter1d(slab_hist, slab_sigma) / area / w_bin
    c_den = gaussian_filter1d(c_hist, c_sigma) / area / w_bin
    return slab_den, c_den


def density_profile(traj, hist_range, bins, gauss_width_slab, gauss_width_C,
                    w_bin, num_surf, return_hist=False,
                    slab_sym='Cu', layer_syms=None):
    """
    Compute Gaussian-smoothed density profiles directly from a trajectory.

    This is a high-level wrapper around :func:`trajectory_histogram_full`
    and :func:`smooth_density_two_species`.

    Parameters
    ----------
    traj : list of ase.Atoms
        MD trajectory.
    hist_range : tuple of float
        (z_min, z_max) histogram range [Angstrom].
    bins : int
        Number of histogram bins.
    gauss_width_slab : float
        Gaussian smoothing width for the slab species [Angstrom].
    gauss_width_C : float
        Gaussian smoothing width for the layer species [Angstrom].
    w_bin : float
        Bin width [Angstrom].
    num_surf : int
        Number of topmost slab atoms used for the surface-distance estimate.
    return_hist : bool, optional
        If True, also return the raw histograms. Default: False.
    slab_sym : str, optional
        Chemical symbol of the slab species. Default: ``'Cu'``.
    layer_syms : list of str, optional
        Chemical symbols of the layer species. Default: ``['C']``.

    Returns
    -------
    slab_den : np.ndarray
        Smoothed slab density [atoms/Angstrom^3].
    c_den : np.ndarray
        Smoothed layer density [atoms/Angstrom^3].
    zs : np.ndarray
        z-coordinates [Angstrom].
    c_mean : float
        Average layer z-position [Angstrom].
    s_dist : float
        Average layer–slab-surface distance [Angstrom].
    slab_hist : np.ndarray
        Raw slab histogram (only when *return_hist* is True).
    c_hist : np.ndarray
        Raw layer histogram (only when *return_hist* is True).
    """
    if layer_syms is None:
        layer_syms = ['C']

    c_hist, slab_hist, c_mean, s_dist, area = trajectory_histogram_full(
        traj, hist_range, bins, num_surf, slab_sym, layer_syms
    )
    slab_den, c_den = smooth_density_two_species(
        slab_hist, c_hist, gauss_width_slab, gauss_width_C, w_bin, area
    )
    zs = np.arange(hist_range[0], hist_range[1], w_bin)

    if return_hist:
        return slab_den, c_den, zs, c_mean, s_dist, slab_hist, c_hist
    return slab_den, c_den, zs, c_mean, s_dist
