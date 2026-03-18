"""
lmcat.io — trajectory reading and atom counting utilities.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def count_element(atoms, ele):
    """
    Count the number of atoms of a given element in an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    ele : str
        Element symbol, e.g. ``'Cu'``, ``'C'``.

    Returns
    -------
    int
        Number of atoms with chemical symbol *ele*.
    """
    return sum(1 for s in atoms.get_chemical_symbols() if s == ele)


def layer_slab_distances(traj, num_surf, ele_M=None, ele_layer=None):
    """
    Compute the average vertical distance between an adsorbed layer and the
    topmost surface atoms of the slab.

    Parameters
    ----------
    traj : list of ase.Atoms
        MD trajectory (list of snapshots).
    num_surf : int
        Number of topmost slab atoms used to define the surface plane.
    ele_M : list of str, optional
        Chemical symbols of the slab (metal) atoms. Default: ``['Cu']``.
    ele_layer : list of str, optional
        Chemical symbols of the adsorbed-layer atoms. Default: ``['C']``.

    Returns
    -------
    s_dist : float
        Average distance between the mean layer z-position and the mean of
        the *num_surf* topmost slab atoms [Angstrom].
    t_dist : float
        Average distance between the mean layer z-position and the mean of
        all slab atom z-positions [Angstrom].

    Notes
    -----
    Only atoms whose symbol is in *ele_M* contribute to the slab positions.
    A previous version of this function collected *all* atoms for the slab,
    which was a bug (fixed here).
    """
    if ele_M is None:
        ele_M = ['Cu']
    if ele_layer is None:
        ele_layer = ['C']

    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    c_indice = [i for i, s in enumerate(symbols) if s in ele_layer]
    cu_indice = [i for i, s in enumerate(symbols) if s in ele_M]  # bug fix: was `if symbols[i]`

    s_dist = 0.0
    t_dist = 0.0
    c_mean = 0.0
    c_z_pos = np.array([])
    c_z_std = 0.0

    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:, 2]
        z1 = zpos[c_indice]
        z2 = zpos[cu_indice]
        c_z_pos = np.concatenate([c_z_pos, z1])
        c_z_std += np.std(z1)
        c_mean += z1.mean()
        surf_zs = z2[np.argsort(z2)[-num_surf:]]
        s_dist += z1.mean() - surf_zs.mean()
        t_dist += z1.mean() - z2.mean()

    n = len(traj)
    c_mean /= n
    c_z_std /= n
    s_dist /= n
    t_dist /= n

    logger.info("Average layer height: %.4f Angstrom", c_mean)
    logger.info("Layer z std (avg over snapshots): %.4f Angstrom", c_z_std)
    logger.info("Layer z std (all snapshots): %.4f Angstrom", np.std(c_z_pos))
    logger.info("s_dist (layer mean - top slab atoms mean): %.4f Angstrom", s_dist)
    logger.info("t_dist (layer mean - all slab atoms mean): %.4f Angstrom", t_dist)

    return s_dist, t_dist
