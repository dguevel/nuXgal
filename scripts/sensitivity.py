from argparse import ArgumentParser
import json
import os

import pandas as pd
import numpy as np
import csky as cy
from tqdm import tqdm
import matplotlib.pyplot as plt
import histlite as hl
import scipy.stats

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal import Defaults

def plot_chi2_distribution(data, outputdir, ninj, ebinmin, ebinmax):
    fig, ax = plt.subplots(dpi=300)
    ax.hist([i[0] for i in data['chi_square'][data['n_inj'] == ninj]], bins=np.arange(0, 900, 10), density=True, label=r'Trial $\chi^2$')
    dof = data['dof'][0]
    print(dof)
    chi2 = scipy.stats.chi2(df=dof)
    x = np.arange(0, 1000)
    ax.step(x, chi2.pdf(x), label='$\chi^2$ PDF with {:d} DoF'.format(dof))
    ax.legend()
    ax.set_xlim(0, 900)
    elo = 10**Defaults.map_logE_edge[ebinmin]
    ehi = 10**Defaults.map_logE_edge[ebinmax]
    ax.set_title('Cross correlation fit bias: {0:.1f}-{1:.1f} TeV'.format(elo, ehi))
    ax.set_xlabel(r'$\chi^2$')
    fname = os.path.join(outputdir, 'unwise_chi2_dist_ebin{}.png'.format(ebinmin))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')

def plot_bg_TS(data, outputdir):
    """Plot the background TS distribution."""
    TS = data['TS'][data['n_inj'] == 0]
    ebin = data['ebinmin'][0]
    logemin = Defaults.map_logE_edge[ebin] - 3
    logemax = Defaults.map_logE_edge[ebin + 1] - 3
    title = 'unWISE-2MASS TS Distribution: {0:d}-{1:d} TeV'.format(
        int(10**logemin), int(10**(logemax)))

    b = cy.dists.Chi2TSD(TS)
    fig, ax = plt.subplots(dpi=300)
    h = b.get_hist(bins=30)
    hl.plot1d(ax, h, crosses=True,
              label='{} bg trials'.format(b.n_total))
    x = h.centers[0]
    norm = h.integrate().values
    #ax.semilogy(x, norm * b.pdf(x), lw=1, ls='--',
    #            label=r'$\chi^2[{:.2f}\text{{dof}},\ \eta={:.3f}]$'.format(b.ndof, b.eta))
    ax.semilogy(x, norm * b.pdf(x), lw=1, ls='--',
        label=r'$\chi^2[{:.2f}$ dof, $\eta={:.3f}]$'.format(b.ndof, b.eta))
    ax.set_xlabel(r'TS')
    ax.set_ylabel(r'number of trials')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(outputdir, 'unwise_TS_dist_ebin{}.png'.format(ebin))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')
    return b

def load_data(files):
    """Load data from json files into a pandas DataFrame."""
    data = []
    for f in tqdm(files):
        with open(f, 'r') as fp:
            data.extend(json.load(fp))

    df = pd.DataFrame(data)
    return df

def plot_sensitivity(sens, disc, ns, gamma=2.0, label='template', outputdir='.', ebin=0):
    """Plot sensitivity and discovery potential."""
    fig, ax = plt.subplots(dpi=100)
    ax.plot(ns, sens, label='Sensitivity')
    ax.plot(ns, disc, label='Discovery Potential')
    #ax.set_xlabel('Flux injected [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$] at 100 TeV')
    ax.set_xlabel(r'$f_{inj}$')
    ax.set_ylabel('Fraction of trials with TS > threshold')
    ax.axhline(0.9, alpha=.5, c='black')
    ax.axhline(0.5, alpha=.5, c='black')
    ax.set_title(label + ': $\gamma=' + str(gamma) + '$')
    ax.legend()
    fname = os.path.join(outputdir, 'unwise_TS_sensitivity_ebin{}.png'.format(ebin))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')

def plot_fit_bias(df, outputdir, ebin=0):
    fig, ax = plt.subplots(dpi=100)
    n_inj = np.sort(np.unique(df['n_inj']))

    logemin = Defaults.map_logE_edge[ebin]
    logemax = Defaults.map_logE_edge[ebin + 1]
    x = [np.mean(df['n_inj'][df['n_inj'] == fi]) for fi in n_inj]
    y = np.array([np.mean([i[0] for i in df['n_fit'][df['n_inj'] == fi]]) for fi in n_inj])
    yerr = np.array([np.std([i[0] for i in df['n_fit'][df['n_inj'] == fi]]) for fi in n_inj])
    ax.fill_between(x, y-yerr, y+yerr, alpha=.5)
    ax.plot(x, y)
    ax.plot(x, x, '--', c='C0')
    ax.set_xlabel('Neutrinos injected')
    ax.set_ylabel('Neutrinos fit')
    ax.set_title('Cross correlation fit bias: {0:d}-{1:d} TeV'.format(int(10**ebin), int(10**(ebin+1))))
    fname = os.path.join(outputdir, 'unwise_fit_bias_ebin{}.png'.format(ebin))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')


def main():
    """Calculate sensitivity, discovery potential, fit bias, 
    and test statistic distribution of cross correlation trials"""
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='Input files')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--ebinmin', help='Minimum energy bin', type=int)
    parser.add_argument('--ebinmax', help='Maximum energy bin', type=int)
    parser.add_argument('--gamma', help='Spectral index', type=float, default=2.5)

    args = parser.parse_args()

    # load data
    print('Reading files...')
    df = load_data(args.input)

    if not args.ebinmin and not args.ebinmax:
        ebinmin = np.unique(df['ebinmin'])
        ebinmax = np.unique(df['ebinmax'])
        if len(ebinmin) != 1 or len(ebinmax) != 1:
            raise IOError('Multiple energy bins found in input files. Please choose one.')
        else:
            ebinmin = ebinmin[0]
            ebinmax = ebinmax[0]
    else:
        ebinmin = args.ebinmin
        ebinmax = args.ebinmax

    # plot background TS distribution
    plot_bg_TS(df, args.output)

    # calculate sensitivity and discovery potential
    b = cy.dists.Chi2TSD(df['TS'][df['n_inj'] == 0])
    n_inj = np.sort(np.unique(df['n_inj']))
    sens = [np.mean(df['TS'][df['n_inj'] == fi] > b.median()) for fi in n_inj]
    disc = [np.mean(df['TS'][df['n_inj'] == fi] > b.isf_nsigma(5.)) for fi in n_inj]
    ebin = df['ebinmin'][0]
    plot_sensitivity(sens, disc, n_inj, label=r'Counts ($l>1$)', gamma=args.gamma, outputdir=args.output, ebin=ebin)

    plot_fit_bias(df, args.output, ebin=ebin)

    plot_chi2_distribution(df, args.output, 0, ebinmin, ebinmax)
    ninj = {0:5000, 1:5000, 2:500, 3:50}
    plot_chi2_distribution(df, args.output, ninj, ebinmin, ebinmax)

    llh = Likelihood(
        'ps_v4',
        'unWISE_z=0.4',
        gamma=args.gamma,
        Ebinmin=ebinmin,
        Ebinmax=ebinmax,
        lmin=0)

    trials = {}
    for n in n_inj:
        trials[n] = df['TS'][df['n_inj'] == n]

    # do the sensitivity calculation using the csky function from trials
    sensitivity = llh.event_generator.trial_runner.find_n_sig(b.median(), 0.9, tss=trials)
    sens_flux = llh.event_generator.trial_runner.to_dNdE(sensitivity['n_sig'], E0=1e5, gamma=2.5) / (4 * np.pi * llh.f_sky)
    print('Sensitivity: ', sens_flux)

    # do the discovery potential calculation using the csky function from trials
    discovery = llh.event_generator.trial_runner.find_n_sig(b.isf_nsigma(5.), 0.5, tss=trials)
    disc_flux = llh.event_generator.trial_runner.to_dNdE(discovery['n_sig'], E0=1e5, gamma=2.5) / (4 * np.pi * llh.f_sky)
    print('Discovery potential: ', disc_flux)




if __name__ == '__main__':
    main()
