"""
lmcat.fitting — erfc/erf fitting to locate Gibbs dividing surfaces.
"""
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erfc, erf
import scipy.optimize

logger = logging.getLogger(__name__)


def _double_erf_sym(z, rho, z1, z2, sigma):
    """
    Symmetric double-erf model for a slab density profile.

    Parameters
    ----------
    z : array-like
        z-coordinates [Angstrom].
    rho : float
        Bulk density [atoms/Angstrom^3].
    z1 : float
        Position of the lower interface [Angstrom].
    z2 : float
        Position of the upper interface [Angstrom].
    sigma : float
        Interface width parameter [Angstrom].

    Returns
    -------
    np.ndarray
        Model density at each *z* value.
    """
    arg1 = (z - z1) / (np.sqrt(2) * sigma)
    arg2 = (z - z2) / (np.sqrt(2) * sigma)
    return 0.5 * rho * (erf(arg1) - erf(arg2))


def fit_interface(slab_prof, slab_sigma, sample_range=None, zs=None,
                  opt_sigma=False, dz=10, ele_M=29):
    """
    Find the Gibbs dividing surface position by fitting an erfc to the density
    profile.

    Parameters
    ----------
    slab_prof : np.ndarray
        Normalized density profile, shape (N,), units [atoms/Angstrom^3].
    slab_sigma : float
        Gaussian smoothing width used to generate *slab_prof* [Angstrom].
        Used as the fixed interface width when *opt_sigma* is False.
    sample_range : np.ndarray, optional
        z-values over which to perform the fit [Angstrom].
        Default: ``np.linspace(30, 50, 300)``.
    zs : np.ndarray, optional
        z-coordinates matching *slab_prof*. Default: ``np.arange(0, 60, 0.1)``.
    opt_sigma : bool, optional
        If True, also optimize the interface width sigma. Default: False.
    dz : float, optional
        Half-width of the fitting window around the half-max point [Angstrom].
        Default: 10.
    ele_M : float, optional
        Atomic number (used as multiplicative factor to convert normalized
        density to counts). Default: 29 (copper).

    Returns
    -------
    mu : float
        Gibbs dividing surface position [Angstrom].
    fit_erf : np.ndarray
        Array of shape (M, 2) with columns [z, model_density], covering
        *sample_range*.

    Notes
    -----
    The Gibbs dividing surface is the position *mu* in::

        rho(z) = factor * erfc((z - mu) / (sigma * sqrt(2)))

    Global optimization is performed with ``scipy.optimize.shgo`` seeded
    with ``seed=42`` for reproducibility.
    """
    if sample_range is None:
        sample_range = np.linspace(30, 50, 300)
    if zs is None:
        zs = np.arange(0, 60, 0.1)

    slab_lin = interp1d(zs, slab_prof * ele_M, kind='linear',
                        bounds_error=False, fill_value=0.0)
    lin_prof = slab_lin(sample_range)

    half_peak = lin_prof.max() / 2.0
    ind_half = np.where(lin_prof > half_peak)[0][-1]
    x_half = sample_range[ind_half]
    logger.info("Half Max position: %.4f Angstrom", x_half)

    xs = np.arange(x_half - dz, x_half + dz, 0.01)

    def error(args):
        ref = slab_lin(xs)
        if opt_sigma:
            factor, mu, s = args
            exp = factor * erfc((xs - mu) / s / np.sqrt(2))
        else:
            factor, mu = args
            exp = factor * erfc((xs - mu) / slab_sigma / np.sqrt(2))
        return np.sum((ref - exp) ** 2) * (xs[1] - xs[0])

    if opt_sigma:
        opt = scipy.optimize.shgo(
            error,
            bounds=[(0.1, lin_prof.max()), (xs[0], xs[-1]),
                    (0.1, max(5, 2 * slab_sigma))],
            options={'seed': 42},
        )
    else:
        opt = scipy.optimize.shgo(
            error,
            bounds=[(0.1, lin_prof.max()), (xs[0], xs[-1])],
            options={'seed': 42},
        )

    logger.info("Optimization results for interface fit:")
    logger.info("  prefactor: %.6f", opt.x[0])
    logger.info("  mu (interface position): %.6f Angstrom", opt.x[1])
    if opt_sigma:
        logger.info("  sigma: %.6f Angstrom", opt.x[2])
    logger.debug("%s", opt)

    if opt_sigma:
        factor, mu, s = opt.x
        exp_prof = factor * erfc((sample_range - mu) / s / np.sqrt(2))
    else:
        factor, mu = opt.x
        exp_prof = factor * erfc((sample_range - mu) / slab_sigma / np.sqrt(2))

    fit_erf = np.array([sample_range, exp_prof / ele_M]).T
    return opt.x[1], fit_erf


def fit_slab_interfaces(zs, den, sigma=None):
    """
    Fit a symmetric double-erf model to a slab density profile to extract
    interface positions and slab thickness.

    Parameters
    ----------
    zs : np.ndarray
        z-coordinates [Angstrom], shape (N,).
    den : np.ndarray
        Density profile [atoms/Angstrom^3], shape (N,).
    sigma : float or None, optional
        If given, fix the interface width to this value [Angstrom].
        If None (default), sigma is a free fit parameter.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``'popt'`` – optimal fit parameters (array).
        * ``'pcov'`` – covariance matrix of the fit parameters.
        * ``'rho'``  – fitted bulk density [atoms/Angstrom^3].
        * ``'z1'``   – lower interface position [Angstrom].
        * ``'z2'``   – upper interface position [Angstrom].
        * ``'sigma'`` – interface width [Angstrom] (only when *sigma* is None).

    Notes
    -----
    The model is::

        rho(z) = 0.5 * rho * (erf((z - z1) / (sqrt(2) * sigma))
                             - erf((z - z2) / (sqrt(2) * sigma)))

    The slab thickness is ``z2 - z1``.
    """
    rho_0 = den.max()
    drdz = np.gradient(den, zs)
    z1_0 = zs[np.argmax(drdz)]   # rising edge
    z2_0 = zs[np.argmin(drdz)]   # falling edge

    if sigma is None:           # bug fix: was `== None`
        model = _double_erf_sym
        p0 = (rho_0, z1_0, z2_0, 1)
    else:
        p0 = (rho_0, z1_0, z2_0)

        def model(z, rho, z1, z2):
            arg1 = (z - z1) / (np.sqrt(2) * sigma)
            arg2 = (z - z2) / (np.sqrt(2) * sigma)
            return 0.5 * rho * (erf(arg1) - erf(arg2))

    popt, pcov = curve_fit(model, zs, den, p0=p0)

    res = {'popt': popt, 'pcov': pcov}
    if sigma is None:           # bug fix: was `== None`
        rho, z1, z2, sigma_new = popt
        res.update({'rho': rho, 'z1': z1, 'z2': z2, 'sigma': sigma_new})  # bug fix: was 'simga'
    else:
        rho, z1, z2 = popt
        res.update({'rho': rho, 'z1': z1, 'z2': z2})

    return res
