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
import json

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
    elo = 10**(Defaults.map_logE_edge[ebinmin] - 3)
    ehi = 10**(Defaults.map_logE_edge[ebinmax] - 3)
    ax.set_title(r'$\chi^2$: {0:.1f}-{1:.1f} TeV'.format(elo, ehi))
    ax.set_xlabel(r'$\chi^2$')
    fname = os.path.join(outputdir, 'unwise_chi2_dist_ebin{}_ninj{}.png'.format(ebinmin, ninj))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')

def plot_bg_TS(data, outputdir):
    """Plot the background TS distribution."""
    TS = data['TS'][data['n_inj'] == 0]
    ebin = data['ebinmin'][0]
    logemin = Defaults.map_logE_edge[ebin] - 3
    logemax = Defaults.map_logE_edge[ebin + 1] - 3
    title = 'unWISE-2MASS TS Distribution: {0:.1f}-{1:.1f} TeV'.format(
        10**logemin, 10**(logemax))

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

    logemin = Defaults.map_logE_edge[ebin] - 3
    logemax = Defaults.map_logE_edge[ebin + 1] - 3
    x = [np.mean(df['n_inj'][df['n_inj'] == fi]) for fi in n_inj]
    y = np.array([np.median([i[0] for i in df['n_fit'][df['n_inj'] == fi]]) for fi in n_inj])
    #yerr = np.array([np.std([i[0] for i in df['n_fit'][df['n_inj'] == fi]]) for fi in n_inj])
    #ax.fill_between(x, y-yerr, y+yerr, alpha=.5)
    yerr = np.array([np.percentile([i[0] for i in df['n_fit'][df['n_inj'] == fi]], [32, 68]) for fi in n_inj]).T
    ax.fill_between(x, yerr[0], yerr[1], alpha=.5)
    ax.plot(x, y)
    ax.plot(x, x, '--', c='C0')
    ax.set_xlabel('Neutrinos injected')
    ax.set_ylabel('Neutrinos fit')
    ax.set_title('Cross correlation fit bias: {0:.1f}-{1:.1f} TeV'.format(10**logemin, 10**logemax))
    fname = os.path.join(outputdir, 'unwise_fit_bias_ebin{}.png'.format(ebin))
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')


def find_n_sig(ts_dict, fraction, threshold):
    # Sort the dictionary keys
    sorted_keys = sorted(ts_dict.keys())

    # Calculate the fraction of test statistics that pass the threshold for each signal level
    fractions_passing_threshold = [np.sum(np.array(ts_dict[key]) > threshold) / len(ts_dict[key]) for key in sorted_keys]

    # Create an interpolation function
    interp_func = scipy.interpolate.interp1d(fractions_passing_threshold, sorted_keys, fill_value='extrapolate')

    # Find the signal level at which the given fraction of test statistics passes the threshold
    signal_level = interp_func(fraction)

    return signal_level

def make_ts_dict(df, ts_key='TS'):
    n_inj = np.sort(np.unique(df['n_inj']))
    ts_dict = {}

    for n in n_inj:
        ts_dict[n] = np.array(df[ts_key][df['n_inj'] == n])

    return ts_dict

def bootstrap_array(arr):
    return np.random.choice(arr, size=len(arr), replace=True)

def sensitivity_err(df, fraction, type, nboot=10, ts_key='TS'):
    sensitivity = np.zeros(nboot)
    for i in tqdm(np.arange(nboot)):
        ts_dict = make_ts_dict(df, ts_key=ts_key)
        for key in ts_dict.keys():
            ts_dict[key] = bootstrap_array(ts_dict[key])
        if type == 'median':
            threshold = cy.dists.Chi2TSD(ts_dict[0]).median()
        elif type == '5sigma':
            threshold = cy.dists.Chi2TSD(ts_dict[0]).isf_nsigma(5.)
        else:
            raise ValueError('Invalid type')
        sensitivity[i] = find_n_sig(ts_dict, fraction, threshold)

    result_dict = {
        'n_sig': np.mean(sensitivity),
        'n_sig_err': np.std(sensitivity),
    }

    return result_dict

def calc_sensitivity(df, trial_runner, template=False, f_sky=1.0, gamma=2.5, ebinmin=0, ebinmax=0):

    if template:
        ts_key = 'template_TS'
    else:
        ts_key = 'TS'

    if template:
        ts_key = 'template_TS'
    else:
        ts_key = 'TS'

    b = cy.dists.Chi2TSD(df[ts_key][df['n_inj'] == 0])

    # do the sensitivity calculation using the csky function from trials
    logemin = Defaults.map_logE_edge[ebinmin]
    logemax = Defaults.map_logE_edge[ebinmax]
    emid = 10 ** ((logemin + logemax) / 2) # in GeV
    #sensitivity = trial_runner.find_n_sig(b.median(), 0.9, tss=trials, tol=1)
    sensitivity = sensitivity_err(df, 0.9, 'median', ts_key=ts_key)
    sens_n_sig = sensitivity['n_sig']
    sens_flux = trial_runner.to_dNdE(sensitivity['n_sig'], E0=emid, gamma=gamma) / (4 * np.pi * f_sky)
    sens_flux_err = trial_runner.to_dNdE(sensitivity['n_sig_err'], E0=emid, gamma=gamma) / (4 * np.pi * f_sky)
    print('Sensitivity: ', sens_flux)

    # do the discovery potential calculation using the csky function from trials
    if np.all(df[ts_key] < b.isf_nsigma(5.)):
        print('No trials with TS > 5 sigma. Setting discovery potential to 0.')
        disc_flux = 0.0
        disc_n_sig = 0.0
    else:
        discovery = sensitivity_err(df, 0.5, type='5sigma', ts_key=ts_key)
        disc_flux = trial_runner.to_dNdE(discovery['n_sig'], E0=emid, gamma=gamma) / (4 * np.pi * f_sky)
        disc_flux_err = trial_runner.to_dNdE(discovery['n_sig_err'], E0=emid, gamma=gamma) / (4 * np.pi * f_sky)
        disc_n_sig = discovery['n_sig']
        if np.isinf(disc_n_sig):
            disc_n_sig = 0.0
            disc_flux = 0.0
    print('Discovery potential: ', disc_flux)

    output = {
        'sens_flux': sens_flux,
        'sens_flux_err': sens_flux_err,
        'sens_n_sig': sens_n_sig,
        'sens_n_sig_err': sensitivity['n_sig_err'],
        'disc_flux': disc_flux,
        'disc_flux_err': disc_flux_err,
        'disc_n_sig': disc_n_sig,
        'disc_n_sig_err': discovery['n_sig_err'],
        'E0': emid,
    }

    return output

def main():
    """Calculate sensitivity, discovery potential, fit bias, 
    and test statistic distribution of cross correlation trials"""
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='Input files')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--nyr', help='Neutrino data to use', 
                        choices=['ps_v4', 'nt_v5'])
    parser.add_argument('--ebinmin', help='Minimum energy bin', type=int)
    parser.add_argument('--ebinmax', help='Maximum energy bin', type=int)
    parser.add_argument('--gamma', help='Spectral index', type=float, default=2.5)
    parser.add_argument('--analytic-thresholds', action='store_true', help='Use analytic thresholds')

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
    if args.analytic_thresholds:
        sens = [np.mean(df['TS'][df['n_inj'] == fi] > 0) for fi in n_inj]
        disc = [np.mean(df['TS'][df['n_inj'] == fi] > 25) for fi in n_inj]
    else:
        sens = [np.mean(df['TS'][df['n_inj'] == fi] > b.median()) for fi in n_inj]
        disc = [np.mean(df['TS'][df['n_inj'] == fi] > b.isf_nsigma(5.)) for fi in n_inj]
    ebin = df['ebinmin'][0]
    plot_sensitivity(sens, disc, n_inj, label=r'Counts ($l>1$)', gamma=args.gamma, outputdir=args.output, ebin=ebin)

    plot_fit_bias(df, args.output, ebin=ebin)

    plot_chi2_distribution(df, args.output, 0, ebinmin, ebinmax)
    ninj = {0:1000, 1:1000, 2:250, 3:30}[ebinmin]
    plot_chi2_distribution(df, args.output, ninj, ebinmin, ebinmax)

    llh = Likelihood(
        args.nyr,
        'unWISE_z=0.4',
        gamma=args.gamma,
        Ebinmin=ebinmin,
        Ebinmax=ebinmax,
        lmin=0)

    output_json = os.path.join(args.output, 'sensitivity_ebin{0:d}-{1:d}.json')

    output_json = output_json.format(ebinmin, ebinmax)

    cc_sens = calc_sensitivity(df, llh.event_generator.trial_runner, template=False, f_sky=llh.f_sky, gamma=args.gamma, ebinmin=ebinmin, ebinmax=ebinmax)
    tmp_sens = calc_sensitivity(df, llh.event_generator.trial_runner, template=True, f_sky=llh.f_sky, gamma=args.gamma, ebinmin=ebinmin, ebinmax=ebinmax)

    output_data = {}
    output_data['sens_flux'] = float(cc_sens['sens_flux'])
    output_data['sens_n_sig'] = float(cc_sens['sens_n_sig'])
    output_data['sens_flux_err'] = float(cc_sens['sens_flux_err'])
    output_data['sens_n_sig_err'] = float(cc_sens['sens_n_sig_err'])
    output_data['disc_flux'] = float(cc_sens['disc_flux'])
    output_data['disc_n_sig'] = float(cc_sens['disc_n_sig'])
    output_data['disc_flux_err'] = float(cc_sens['disc_flux_err'])
    output_data['disc_n_sig_err'] = float(cc_sens['disc_n_sig_err'])
    output_data['E0'] = float(cc_sens['E0'])
    output_data['template_sens_flux'] = float(tmp_sens['sens_flux'])
    output_data['template_sens_n_sig'] = float(tmp_sens['sens_n_sig'])
    output_data['template_sens_flux_err'] = float(tmp_sens['sens_flux_err'])
    output_data['template_sens_n_sig_err'] = float(tmp_sens['sens_n_sig_err'])
    output_data['template_disc_flux'] = float(tmp_sens['disc_flux'])
    output_data['template_disc_n_sig'] = float(tmp_sens['disc_n_sig'])
    output_data['template_disc_flux_err'] = float(tmp_sens['disc_flux_err'])
    output_data['template_disc_n_sig_err'] = float(tmp_sens['disc_n_sig_err'])
    output_data['ebinmin'] = int(ebinmin)
    output_data['ebinmax'] = int(ebinmax)
    output_data['logemin'] = float(Defaults.map_logE_edge[ebinmin])
    output_data['logemax'] = float(Defaults.map_logE_edge[ebinmax])
    output_data['gamma'] = float(args.gamma)
    output_data['munu_ns_spl'] = llh.event_generator.trial_runner.to_ns(
        (1e5)**2 * 1.44e-18 * (4 * np.pi * llh.f_sky),
        E0=1e5,
        gamma=2.37
    )
    output_data['munu_ns_spl_std'] = llh.event_generator.trial_runner.to_ns(
        (1e5)**2 * 0.25e-18 * (4 * np.pi * llh.f_sky),
        E0=1e5,
        gamma=2.37
    )

    with open(output_json, 'w') as fp:
        json.dump(output_data, fp, indent=4)


if __name__ == '__main__':
    main()
