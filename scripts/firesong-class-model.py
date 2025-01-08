#/usr/bin/env python3

from argparse import ArgumentParser
import os

import numpy as np
from firesong.Firesong import firesong_simulation
import pyccl as ccl
from scipy.interpolate import interp1d



def load_firesong(filename):
    ra, dec, z, flux = np.loadtxt(filename, skiprows=6, delimiter=' ').T
    return {'ra': ra, 'dec': dec, 'z': z, 'flux': flux}

def compute_cls_class(zmean, zwidth, dndz_fname):

    LambdaCDM = Class()
    # pass input parameters
    LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk,nCl','lensing':'yes','P_k_max_1/Mpc':3.0})
    LambdaCDM.set({'l_max_lss': '400'})
    LambdaCDM.set({'selection': 'gaussian', 'selection_mean': str(zmean), 'selection_width': str(zwidth)})
    LambdaCDM.set({'number_count_contributions': 'density, rsd, lensing, gr'})
    #LambdaCDM.set({'non_linear': 'halofit'})
    LambdaCDM.set({'non_linear': 'hmcode'})
    LambdaCDM.set({'dNdz_evolution': dndz_fname})

    # run class
    LambdaCDM.compute()
    density_cl = LambdaCDM.density_cl()['dd'][0]
    return density_cl

def compute_cls_ccl(zmean, zwidth, z_nu, dndz_nu, b_nu=1):
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0, 8, 100)
    dNdz = np.exp(-0.5 * (z - zmean)**2 / zwidth**2)
    z = np.loadtxt('/home/dguevel/git/nuXgal/nb/unWISE-2MASS-GAMA-filtered.csv', skiprows=1, delimiter=',', dtype=str)[:, 9].astype(float)
    dNdz, z = np.histogram(z, bins=np.arange(0, 8, 0.025), range=(0, 4))
    z = z[:-1]

    unwise_tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, 1.23*np.ones_like(z)))
    icecube_tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_nu, dndz_nu), bias=(z, b_nu*np.ones_like(z)))

    ell = np.arange(384)
    crosscorr_cl = ccl.angular_cl(cosmo, unwise_tracer, icecube_tracer, ell)
    autocorr_cl = ccl.angular_cl(cosmo, unwise_tracer, unwise_tracer, ell)
    return crosscorr_cl, autocorr_cl

def main():
    parser = ArgumentParser(description='Compute cross-correlation of a neutrino source redshift distribution and unWISE-2MASS galaxies')
    parser.add_argument('neutrino_dndz', help='Model to use for the neutrino source redshift distribution', choices=['powerlaw', 'sfr', 'no_evolution'])
    parser.add_argument('output_dir', help='Output directory for the simulation')
    parser.add_argument('-k', help='Neutrino source dndz power law index', type=float, default=-2)
    parser.add_argument('-xi', help='Neutrino source dndz power law cutoff', type=float, default=-1)
    parser.add_argument('-b', help='Neutrino source bias', type=float, default=1)

    args = parser.parse_args()

    if args.neutrino_dndz == 'sfr':
        fname = 'firesong_MD2014SFR.out'
        firesong_simulation(args.output_dir, filename=fname, density=1e-5, Evolution='MD2014SFR', zmax=8.0, fluxnorm=1.36e-8, index=2.28, LF='SC')
    elif args.neutrino_dndz == 'no_evolution':
        fname = 'firesong_no_evolution.out'
        firesong_simulation(args.output_dir, filename=fname, density=1e-5, Evolution='NoEvolution', zmax=8.0, fluxnorm=1.36e-8, index=2.28, LF='SC')
    else:
        fname = 'firesong_k{}_xi{}.out'.format(args.k, args.xi)
        firesong_simulation(args.output_dir, filename=fname, density=1e-5, Evolution='PowerLaw', k=args.k, xi=args.xi, zmax=8.0, fluxnorm=1.36e-8, index=2.37, LF='SC')

    firesong_sim = load_firesong(fname)
    zbins = np.arange(0, 8, .025)
    dndz = np.histogram(firesong_sim['z'], bins=zbins, density=True, weights=firesong_sim['flux'])[0]
    from scipy.integrate import trapezoid

    zmid = zbins[:-1]
    dndz_fname = os.path.join(args.output_dir, fname.replace('firesong', 'dndz'))
    np.savetxt(dndz_fname, np.column_stack((zmid, dndz)))

    crosscorr_cl, autocorr_cl = compute_cls_ccl(0.12, 0.072, zmid, dndz, b_nu=args.b)
    mean = np.mean(crosscorr_cl[10:] / autocorr_cl[10:])
    std = np.std(crosscorr_cl[10:] / autocorr_cl[10:])
    f_astro = 0.013
    print('Integral ratio: {0:1.4f} {1:1.4f} {2:1.4f}'.format(mean, std, mean * f_astro))
    idx = zmid < 0.3
    fcorr = trapezoid(dndz[idx], zmid[idx])
    print('fcorr approximation: {0:1.4f} {1:1.4f}'.format(fcorr, fcorr * f_astro))
    

    zbins = np.arange(0, 8, .025)
    dndz = np.histogram(firesong_sim['z'], bins=zbins, density=True)[0]
    zmid = (zbins[:-1] + zbins[1:]) / 2
    zmid = np.concatenate([[0], zmid])
    dndz = np.concatenate([[0], dndz])
    cdf = np.cumsum(dndz)
    cdf /= cdf[-1]
    inv_cdf = interp1d(cdf, zmid)
    #print('median sources redshift', inv_cdf(.5))

    zbins = np.arange(0, 8, .025)
    dndz = np.histogram(firesong_sim['z'], bins=zbins, density=True, weights=firesong_sim['flux'])[0]
    zmid = (zbins[:-1] + zbins[1:]) / 2
    zmid = np.concatenate([[0], zmid])
    dndz = np.concatenate([[0], dndz])
    cdf = np.cumsum(dndz)
    cdf /= cdf[-1]
    inv_cdf = interp1d(cdf, zmid)
    #print('median sources redshift', inv_cdf(.5))


if __name__ == '__main__':
    main()