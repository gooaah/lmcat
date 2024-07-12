# All the functions for DirectSepa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks
from scipy.special import erfc
import scipy.optimize
import ase.io
import matplotlib.pyplot as plt
import sys, os
from copy import deepcopy



def get_fwhm_density(dcu):
    # get fwhm point - larger than x=30
    dcu30 = deepcopy(dcu)
    # focus on half of density (avoid fwhm on "wrong side")
    dcu30[np.where(dcu30[:,0] < 30)[0],1] = 0.0
    dmax = dcu30[:,1].max()
    dcu30[np.where(dcu30[:,1] > dmax/2.0)[0],1] = 0.0
    ind_fwhm = dcu30[:,1].argmax()

    xy_fwhm = dcu[ind_fwhm,:]

    return ind_fwhm, xy_fwhm

def get_zero_2d(sp2d):
    i_infl, x_infl = [], []
    ### identify inflection points through changes in sign
    for i in range(1, sp2d[:,1].size):
        if (sp2d[i,1] * sp2d[i-1,1]) < 0:
            i_infl.append(i)
            x_infl.append(sp2d[i,0])
    return(np.array(i_infl), np.array(x_infl))


def align(cu_prof, c_prof, exp_prof, sample_range=np.linspace(20,50,300), zs=np.arange(0,60,0.1)):
    """
    Note: only for copper
    """
    #zs = np.arange(0,60,0.1)
    #zs_C = np.arange(0,60,0.1)
    zs_C = zs[:]
    cu = interp1d(zs, cu_prof * 29, kind='linear', bounds_error = False, fill_value = 0.0)
    c = interp1d(zs_C, c_prof * 6.0, kind='linear', bounds_error = False, fill_value = 0.0)
    def error(args):
        x0, = args
        exp = interp1d(exp_prof[:,0] + x0, exp_prof[:,1], kind='linear')
        xs = exp_prof[:,0] + x0
        ref = cu(xs) + c(xs)
        diff_sq = (ref - exp(xs)) * (ref - exp(xs))
        return(np.sum(diff_sq) * (xs[1]-xs[0]))

    #opt = scipy.optimize.minimize(error, np.array([10.0]))
    opt = scipy.optimize.shgo(error, np.array([(5.0,60.0)]))
    return opt, cu(sample_range), c(sample_range)


def get_ele_prof(slab_prof, c_prof, sample_range=np.linspace(20,50,300), zs=np.arange(0,60,0.1), slab_ele_num=29, layer_ele_num=6):
    """

    """
    #zs = np.arange(0,60,0.1)
    #zs_C = np.arange(0,60,0.1)
    zs_C = zs[:]
    slab = interp1d(zs, slab_prof * slab_ele_num, kind='linear', bounds_error = False, fill_value = 0.0)
    c = interp1d(zs_C, c_prof * layer_ele_num, kind='linear', bounds_error = False, fill_value = 0.0)


    return slab(sample_range), c(sample_range)

def get_inflection(slab_prof, slab_sigma, sample_range=np.linspace(30,50,300),zs=np.arange(0,60,0.1),opt_sigma=False, dz=10, ele_M=29):

    slab_lin = interp1d(zs, slab_prof * ele_M, kind='linear', bounds_error = False, fill_value = 0.0)
    lin_prof = slab_lin(sample_range)


    # print(f"peak : {sample_range[slab_peaks[-1]]}")
    half_peak = lin_prof.max() / 2.
    ind_half = np.where(lin_prof>half_peak)[0][-1]


    x_half = sample_range[ind_half]
    print(f"Half Max: {x_half}")

    xs = np.arange(x_half-dz, x_half+dz, 0.01)



    def error(args):
        ref = slab_lin(xs)
        # prefactor, mean, sigma
        if opt_sigma:
            factor, mu, s = args
            exp = factor*erfc((xs-mu)/s/np.sqrt(2))
        else:
            factor, mu, = args
            exp = factor*erfc((xs-mu)/slab_sigma/np.sqrt(2))
        diff_sq = np.sum((ref-exp)**2) * (xs[1] - xs[0])
        return diff_sq

    if opt_sigma:
        opt = scipy.optimize.shgo(error, bounds=[(0.1,lin_prof.max()),(xs[0],xs[-1]), (0.1,max(5,2*slab_sigma))])
    else:
        opt = scipy.optimize.shgo(error, bounds=[(0.1,lin_prof.max()),(xs[0],xs[-1]),])
    # opt = scipy.optimize.minimize(error, [lin_prof.max()/2, 0.5*(xs[0]+xs[-1]) ])
    # opt = scipy.optimize.shgo(error, bounds=[(0.1,None),(xs[0],xs[1]),(0.5*slab_sigma,2*slab_sigma)])
    print("Optimization results for inflection:")
    print(f"prefactor: {opt.x[0]}")
    print(f"mean: {opt.x[1]}")
    if opt_sigma:
        print(f"sigma: {opt.x[2]}")
    print(opt)

    if opt_sigma:
        factor, mu, s = opt.x
        exp_prof = factor*erfc((sample_range-mu)/s/np.sqrt(2))
    else:
        factor, mu = opt.x
        exp_prof = factor*erfc((sample_range-mu)/slab_sigma/np.sqrt(2))
    # np.savetxt('fit_infl.dat', np.array([xs, slab_lin(xs), exp_prof]).T)
    fit_erf = np.array([sample_range, exp_prof]).T

    return opt.x[1], fit_erf





def get_inflection_old(cu_prof, sample_range=np.linspace(20,50,300),zs=np.arange(0,60,0.1),s=0):

    cu_sp = UnivariateSpline(zs, cu_prof * 29, s=s, k=3)
    cu_lin = interp1d(zs, cu_prof * 29, kind='linear', bounds_error = False, fill_value = 0.0)
    lin_prof = cu_lin(sample_range)
    sp_prof = cu_sp(sample_range)
    cu_peaks = find_peaks(lin_prof)[0]

    print(f"peak index: {cu_peaks[-1]}")
    print(f"peak : {lin_prof[cu_peaks[-1]]}")
    half_peak = lin_prof[cu_peaks[-1]] / 2.
    for i in range(cu_peaks[-1], len(lin_prof)):
        if lin_prof[i] < half_peak:
            ind_half = i
            break

    der2d = cu_sp.derivative(n=2)
    der2d_line = np.array([sample_range, der2d(sample_range)]).T
    i_infl, x_infl = get_zero_2d(der2d_line)
    ii = np.absolute(x_infl - sample_range[ind_half]).argmin()
    ind_infl = i_infl[ii]


    diff_half_infl = sample_range[ind_infl] - sample_range[ind_half]
    print(f"Half max index: {ind_half}")
    print(f"Inflection index: {ind_infl}")
    print(f"Half max: {sample_range[ind_half]}")
    print(f"Inflection: {sample_range[ind_infl]}")
    print('Difference between half max and inflection point: {}'.format(diff_half_infl))



    return sample_range[ind_infl]

def get_num_element(a, ele):
    """
    Calculate the number of element
    """
    symbols = a.get_chemical_symbols()
#    c_indice = [ind for ind in range(len(a0)) if symbols[ind]=='C']
    numE = 0
    for s in symbols:
        if s == ele:
            numE += 1

    return numE

def get_num_carbon(a):
    """
    Calculate the number of carbon atoms
    """
    symbols = a.get_chemical_symbols()
#    c_indice = [ind for ind in range(len(a0)) if symbols[ind]=='C']
    numC = 0
    for s in symbols:
        if s == 'C':
            numC += 1

    return numC

def get_num_copper(a):
    """
    Calculate the number of copper atoms
    """
    symbols = a.get_chemical_symbols()
    #cu_indice = [ind for ind in range(len(a0)) if symbols[ind]=='Cu']
    #numCu = len(cu_indice)
    numCu = 0
    for s in symbols:
        if s == 'Cu':
            numCu += 1

    return numCu

def get_direct_gap(traj, num_surf):
    """
    Get "direct" gap
    """
    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    c_indice = [ind for ind in range(len(a0)) if symbols[ind]=='C']
    cu_indice = [ind for ind in range(len(a0)) if symbols[ind]=='Cu']
    # solid distance
    s_dist = 0
    t_dist = 0 # dist between mean of C and mean of Cu slab
    c_mean = 0
    cu_max = 0
    c_z_pos = np.array([])
    c_z_std = 0
    c_thick = 0

    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:,2]
        z1 = zpos[c_indice]
        c_z_pos = np.concatenate([c_z_pos,z1])
        c_z_std += np.std(z1)
        z2 = zpos[cu_indice]
        # distance between graphene sheet and copper surface
        c_mean += z1.mean()
        surf_zs = z2[np.argsort(z2)[-1*num_surf:]]
        s_dist += z1.mean()- surf_zs.mean()
        t_dist += z1.mean()- z2.mean()

    c_mean /= len(traj)
    c_z_std /= len(traj)
    # only for solid
    #cu_mean /= len(traj)
    s_dist /= len(traj)
    t_dist /= len(traj)
    #dist = c_mean - cu_max

    return s_dist, t_dist

def simple_get_hist(traj, histRange, bins, num_surf, slab_sym):

    # Get area of xy plane; only suitable for orthogonal cell
    cell = traj[0].get_cell()
    area = cell[0,0] * cell[1,1]

    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    c_indice = [ind for ind in range(len(a0)) if symbols[ind]=='C']
    slab_indice = [ind for ind in range(len(a0)) if symbols[ind]==slab_sym]
    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)
    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:,2]
        z1 = zpos[c_indice]
        h1, _ = np.histogram(z1, bins=bins, range=histRange, density=False)
        hist1 += h1
        z2 = zpos[slab_indice]
        h2, _ = np.histogram(z2, bins=bins, range=histRange, density=False)
        hist2 += h2

    # make the unit to be #atoms/(A^3)
    binsize = (histRange[1] - histRange[0])/bins
    c_hist = hist1/len(traj)/area/binsize
    slab_hist = hist2/len(traj)/area/binsize
    # zs = np.arange(histRange[0], histRange[1], w_bin)


    return c_hist, slab_hist


def get_hist(traj, histRange, bins, num_surf, slab_sym='Cu', layer_syms=['C']):
    """
    Get histogram and "direct" gap
    """


    cell = traj[0].get_cell()
    area = cell[0,0] * cell[1,1]

    a0 = traj[0]
    symbols = a0.get_chemical_symbols()
    c_indice = [ind for ind in range(len(a0)) if symbols[ind] in layer_syms]
    slab_indice = [ind for ind in range(len(a0)) if symbols[ind]==slab_sym]

    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)
    # solid distance
    s_dist = 0
    c_mean = 0
    slab_max = 0
    c_z_pos = np.array([])
    c_z_std = 0
    c_thick = 0
    for atms in traj:
        pos = atms.get_positions()
        zpos = pos[:,2]
        z1 = zpos[c_indice]
        c_z_pos = np.concatenate([c_z_pos,z1])
        c_z_std += np.std(z1)
        c_thick += z1.max() - z1.min()
        # z1 = zpos[:numCarbon]
        # z1 = zpos[-numCarbon:]
        h1, _ = np.histogram(z1, bins=bins, range=histRange)
        hist1 += h1
        z2 = zpos[slab_indice]
        # z2 = zpos[numCarbon:]
        # z2 = zpos[:-numCarbon]
        h2, _ = np.histogram(z2, bins=bins, range=histRange)
        hist2 += h2
        # distance between graphene sheet and copper surface
        c_mean += z1.mean()
        slab_max += z2.max()
        # only for solid
        # surface Cu atoms
        surf_zs = z2[np.argsort(z2)[-1*num_surf:]]
        s_dist += z1.mean()- surf_zs.mean()
        #s_dist += z1.mean()- z2[np.arange(7,1344,8)].mean()


    c_mean /= len(traj)
    slab_max /= len(traj)
    c_z_std /= len(traj)
    c_thick /= len(traj)
    # only for solid
    #slab_mean /= len(traj)
    s_dist /= len(traj)
    #dist = c_mean - slab_max
    #
    # print(np.linspace(20,50,100)[89] - slab_max)

    print('average C height: {}'.format(c_mean))
    print('std for carbon layer (all snapshot): {}'.format(np.std(c_z_pos)))
    print('std for carbon layer (average std): {}'.format(c_z_std))
    print('thinkness of carbon layer (max-min): {}'.format(c_thick))
    print('c_mean - slab_max: {}'.format(c_mean - slab_max))
    print('(Only for solid) solid dist: {}'.format(s_dist))


    # ase.io.write('last_cfg.vasp', atms, direct=True)

    c_hist = hist1/len(traj)
    slab_hist = hist2/len(traj)
    # zs = np.arange(histRange[0], histRange[1], w_bin)

    return c_hist, slab_hist, c_mean, s_dist, area

def hist2profile(slab_hist, c_hist, gauss_width_slab, gauss_width_C, w_bin, area):
    """
    From histogram to density profile
    """

    # get smearing parameters
    slab_sigma = gauss_width_slab/w_bin
    c_sigma = gauss_width_C/w_bin


    print('Bin width: {}'.format(w_bin))
    #print('Gaussian width: {}'.format(w_bin*gauss_sigma))
    print('Gaussian width for copper: {}'.format(gauss_width_slab))
    print(slab_sigma)
    print('Gaussian width for carbon: {}'.format(gauss_width_C))
    print(c_sigma)


    # slab_den = gaussian_filter1d(hist2, gauss_sigma) / area / w_bin
    # c_den = gaussian_filter1d(hist1, gauss_sigma) / area / w_bin
    slab_den = gaussian_filter1d(slab_hist, slab_sigma) / area / w_bin
    c_den = gaussian_filter1d(c_hist, c_sigma) / area / w_bin

    return slab_den, c_den



def get_profile(traj, histRange, bins, gauss_width_slab, gauss_width_C, w_bin, num_surf, return_hist=False, slab_sym='Cu', layer_syms=['C']):
    """
    Get density profile from trajectory directly.
    """

    c_hist, slab_hist, c_mean, s_dist, area = get_hist(traj, histRange, bins, num_surf, slab_sym, layer_syms)

    slab_den, c_den = hist2profile(slab_hist, c_hist, gauss_width_slab, gauss_width_C, w_bin, area)

    zs = np.arange(histRange[0], histRange[1], w_bin)

    if return_hist:
        return slab_den, c_den, zs, c_mean, s_dist, slab_hist, c_hist
    else:
        return slab_den, c_den, zs, c_mean, s_dist

def get_distances(slab_prof, c_prof, c_mean, sample_range):
    c_peaks = find_peaks(c_prof)[0]
    cu_peaks = find_peaks(slab_prof)[0]
    if len(cu_peaks) > 0:
        dist_peak = sample_range[c_peaks[-1]] - sample_range[cu_peaks[-1]]
        print('Height of graphene sheet from density: {}'.format(dist_peak))

    half_peak = slab_prof.max() / 2.
    ind_half = np.where(slab_prof>half_peak)[0][-1]
    # half_peak = slab_prof[cu_peaks[-1]] / 2.
    # for i in range(cu_peaks[-1], c_peaks[-1]):
    #     if slab_prof[i] < half_peak:
    #         ind_half = i
    #         break
    # dist_half = sample_range[c_peaks[-1]] - sample_range[ind_half]
    # print('Height of graphene sheet from half peak of density: {}'.format(dist_half))

    if len(cu_peaks) > 0:
        dist_peak_pos = c_mean - sample_range[cu_peaks[-1]]
        print('Height of graphene sheet from slab density and C mean position: {}'.format(dist_peak_pos))


    dist_half_pos = c_mean - sample_range[ind_half]
    print('Height of graphene sheet from half peak of slab and C mean position: {}'.format(dist_half_pos))

    print('c_peak: {}'.format(c_peaks[-1]))
    # print('cu_peak: {}'.format(cu_peaks[-1]))
    print('half cu_peak: {}'.format(ind_half))
    print(sample_range[ind_half])

    return sample_range[ind_half], c_peaks[-1]
