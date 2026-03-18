"""
lmcat.analysis — high-level analysis: profile alignment, half-max density,
interface distances.
"""
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import scipy.optimize

logger = logging.getLogger(__name__)


def align_profiles(cu_prof, c_prof, exp_prof,
                   sample_range=None, zs=None):
    """
    Align an experimental density profile to computed Cu + C density profiles
    by minimizing the squared residual over a rigid z-shift.

    Parameters
    ----------
    cu_prof : np.ndarray
        Normalized Cu density profile, shape (N,) [atoms/Angstrom^3 / atom].
    c_prof : np.ndarray
        Normalized C density profile, shape (N,) [atoms/Angstrom^3 / atom].
    exp_prof : np.ndarray
        Experimental density profile, shape (M, 2), columns [z, density].
    sample_range : np.ndarray, optional
        z-values at which to evaluate the aligned profiles [Angstrom].
        Default: ``np.linspace(20, 50, 300)``.
    zs : np.ndarray, optional
        z-coordinates matching *cu_prof* and *c_prof*. Default:
        ``np.arange(0, 60, 0.1)``.

    Returns
    -------
    opt : scipy.optimize.OptimizeResult
        Optimization result; the optimal z-shift is ``opt.x[0]`` [Angstrom].
    cu_sampled : np.ndarray
        Cu density evaluated at *sample_range*, scaled by atomic number 29
        [atoms/Angstrom^3].
    c_sampled : np.ndarray
        C density evaluated at *sample_range*, scaled by atomic number 6
        [atoms/Angstrom^3].

    Notes
    -----
    Global optimization is performed with ``scipy.optimize.shgo`` seeded
    with ``seed=42`` for reproducibility.
    """
    if sample_range is None:
        sample_range = np.linspace(20, 50, 300)
    if zs is None:
        zs = np.arange(0, 60, 0.1)

    zs_C = zs[:]
    cu = interp1d(zs, cu_prof * 29, kind='linear', bounds_error=False, fill_value=0.0)
    c = interp1d(zs_C, c_prof * 6.0, kind='linear', bounds_error=False, fill_value=0.0)

    def error(args):
        x0, = args
        exp = interp1d(exp_prof[:, 0] + x0, exp_prof[:, 1], kind='linear')
        xs = exp_prof[:, 0] + x0
        ref = cu(xs) + c(xs)
        diff_sq = (ref - exp(xs)) ** 2
        return np.sum(diff_sq) * (xs[1] - xs[0])

    opt = scipy.optimize.shgo(error, np.array([(5.0, 60.0)]),
                              options={'seed': 42})
    return opt, cu(sample_range), c(sample_range)


def element_density(slab_prof, c_prof,
                    sample_range=None, zs=None,
                    slab_ele_num=29, layer_ele_num=6):
    """
    Evaluate element-specific density profiles at a given set of z-positions.

    Parameters
    ----------
    slab_prof : np.ndarray
        Normalized slab density profile [atoms/Angstrom^3 / atom], shape (N,).
    c_prof : np.ndarray
        Normalized layer density profile [atoms/Angstrom^3 / atom], shape (N,).
    sample_range : np.ndarray, optional
        z-values at which to evaluate [Angstrom].
        Default: ``np.linspace(20, 50, 300)``.
    zs : np.ndarray, optional
        z-coordinates matching *slab_prof* and *c_prof*.
        Default: ``np.arange(0, 60, 0.1)``.
    slab_ele_num : float, optional
        Atomic number of the slab element used as a density scale factor.
        Default: 29 (Cu).
    layer_ele_num : float, optional
        Atomic number of the layer element used as a density scale factor.
        Default: 6 (C).

    Returns
    -------
    slab_sampled : np.ndarray
        Slab density at *sample_range* [atoms/Angstrom^3].
    c_sampled : np.ndarray
        Layer density at *sample_range* [atoms/Angstrom^3].
    """
    if sample_range is None:
        sample_range = np.linspace(20, 50, 300)
    if zs is None:
        zs = np.arange(0, 60, 0.1)

    slab = interp1d(zs, slab_prof * slab_ele_num, kind='linear',
                    bounds_error=False, fill_value=0.0)
    c = interp1d(zs, c_prof * layer_ele_num, kind='linear',
                 bounds_error=False, fill_value=0.0)
    return slab(sample_range), c(sample_range)


def density_half_max(dcu, z_min=30.0):
    """
    Find the half-maximum point of a density profile above a z threshold.

    The function looks for the outermost point where the density exceeds half
    of the profile maximum, restricted to ``z >= z_min``.

    Parameters
    ----------
    dcu : np.ndarray
        Density profile, shape (N, 2), columns [z, density]
        [atoms/Angstrom^3].
    z_min : float, optional
        Only consider points with z >= *z_min* [Angstrom]. This avoids
        picking up the half-max on the "wrong" (inner) side of the slab.
        Default: 30.0.

    Returns
    -------
    ind_fwhm : int
        Index into *dcu* of the half-max point.
    xy_fwhm : np.ndarray
        Array [z, density] at the half-max point.
    """
    dcu_work = dcu.copy()
    dcu_work[dcu_work[:, 0] < z_min, 1] = 0.0
    dmax = dcu_work[:, 1].max()
    dcu_work[dcu_work[:, 1] > dmax / 2.0, 1] = 0.0
    ind_fwhm = dcu_work[:, 1].argmax()
    return ind_fwhm, dcu[ind_fwhm, :]


def interface_distances(slab_prof, c_prof, c_mean, sample_range):
    """
    Compute distances between an adsorbed layer and the slab surface using
    peak and half-maximum positions.

    Parameters
    ----------
    slab_prof : np.ndarray
        Slab density profile [atoms/Angstrom^3], shape (N,).
    c_prof : np.ndarray
        Layer density profile [atoms/Angstrom^3], shape (N,).
    c_mean : float
        Time-averaged mean z-position of the layer atoms [Angstrom].
    sample_range : np.ndarray
        z-values matching *slab_prof* and *c_prof* [Angstrom].

    Returns
    -------
    z_half : float
        z-position of the half-maximum of *slab_prof* [Angstrom].  This is
        used as a proxy for the slab surface position.
    c_peak_ind : int
        Index of the last (outermost) peak of *c_prof* in *sample_range*.

    Notes
    -----
    Logged distances:

    * ``dist_peak``: layer peak z − slab peak z (from density peaks).
    * ``dist_peak_pos``: *c_mean* − slab peak z.
    * ``dist_half_pos``: *c_mean* − half-max of slab density.
    """
    c_peaks = find_peaks(c_prof)[0]
    cu_peaks = find_peaks(slab_prof)[0]

    if len(cu_peaks) > 0:
        dist_peak = sample_range[c_peaks[-1]] - sample_range[cu_peaks[-1]]
        logger.info("Height of layer from density peaks: %.4f Angstrom", dist_peak)

    half_peak = slab_prof.max() / 2.0
    ind_half = np.where(slab_prof > half_peak)[0][-1]

    if len(cu_peaks) > 0:
        dist_peak_pos = c_mean - sample_range[cu_peaks[-1]]
        logger.info("Height of layer (c_mean - slab peak): %.4f Angstrom", dist_peak_pos)

    dist_half_pos = c_mean - sample_range[ind_half]
    logger.info("Height of layer (c_mean - slab half-max): %.4f Angstrom", dist_half_pos)
    logger.debug("c_peak index: %d", c_peaks[-1])
    logger.debug("slab half-max index: %d  z=%.4f Angstrom", ind_half, sample_range[ind_half])

    return sample_range[ind_half], c_peaks[-1]
