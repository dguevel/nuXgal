from argparse import ArgumentParser
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csky as cy
from tqdm import tqdm
import histlite as hl
import healpy as hp


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs='+', help="Input file")
    parser.add_argument("-o", "--output", help="Output dirctory")
    args = parser.parse_args()

    df = load_files(args.input)

    plot_bg_TS(df, args.output)
    plot_fit_bias(df, args.output)

    bg = cy.dists.Chi2TSD(df['ts'][df['n_inj'] == 0])
    ns = np.sort(np.unique(df['n_inj']))
    sens = [np.mean(df['ts'][df['n_inj'] == ni] > bg.median()) for ni in ns]
    disc = [np.mean(df['ts'][df['n_inj'] == ni] > bg.isf_nsigma(5.)) for ni in ns]

    plot_sensitivity(sens, disc, ns, gamma=2.5, label='template', outputdir=args.output)


    template = hp.read_map("/home/dguevel/git/nuXgal/data/ancil/unWISE_z=0.4_galaxymap.fits")

    ana = cy.conf.get_analysis(
        cy.selections.repo,
        "version-004-p02",
        cy.selections.PSDataSpecs.ps_v4,
        #dir=cy.utils.ensure_dir("/data/user/dguevel/template/unWISE_v0.4_PS_v4/ana")
    )
    #ana.save("/data/user/dguevel/template/unWISE_v0.4_PS_v4/ana")

    conf = {
        'ana': ana,
        'template': template,
        'flux': cy.hyp.PowerLawFlux(gamma=2.5),
        'sigsub': True,
        'fast_weight': True,
        'dir': cy.utils.ensure_dir("/data/user/dguevel/template/unWISE_v0.4_PS_v4/templates")
    }

    trial_runner = cy.get_trial_runner(conf)
    calc_sensitivity(df, trial_runner, template=True, f_sky=0.45, gamma=2.5)


def load_files(filenames):
    data = []
    for input_file in tqdm(filenames):
        data.append(np.load(input_file, allow_pickle=True))

    data = np.concatenate(data)

    df = pd.DataFrame(data)
    return df


def plot_fit_bias(df, outputdir):
    fig, ax = plt.subplots(dpi=100)
    n_inj = np.sort(np.unique(df['n_inj']))

    x = [np.mean(df['n_inj'][df['n_inj'] == fi]) for fi in n_inj]
    y = np.array([np.median(df['ns'][df['n_inj'] == fi]) for fi in n_inj])
    print(list(zip(x, y)))
    #yerr = np.array([np.std([i[0] for i in df['n_fit'][df['n_inj'] == fi]]) for fi in n_inj])
    #ax.fill_between(x, y-yerr, y+yerr, alpha=.5)
    yerr = np.array([np.percentile(df['ns'][df['n_inj'] == fi], [32, 68]) for fi in n_inj]).T

    ax.fill_between(x, yerr[0], yerr[1], alpha=.5)
    ax.plot(x, y)
    ax.plot(x, x, '--', c='C0')
    ax.set_xlabel('Neutrinos injected')
    ax.set_ylabel('Neutrinos fit')
    ax.set_title('Template fit bias')
    fname = os.path.join(outputdir, 'unwise_fit_bias.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')


def plot_bg_TS(data, outputdir):
    """Plot the background TS distribution."""
    TS = data['ts'][data['n_inj'] == 0]
    title = 'unWISE-2MASS TS Distribution'

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
    fname = os.path.join(outputdir, 'unwise_TS_dist.png')
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

def plot_sensitivity(sens, disc, ns, gamma=2.5, label='template', outputdir='.'):
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
    fname = os.path.join(outputdir, 'unwise_TS_sensitivity.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace('png', 'pdf'), bbox_inches='tight')


def calc_sensitivity(df, trial_runner, template=False, f_sky=1.0, gamma=2.5):
    n_inj = np.sort(np.unique(df['n_inj']))

    trials = {}
    for n in n_inj:
        trials[n] = df['ts'][df['n_inj'] == n]

    b = cy.dists.Chi2TSD(df['ts'][df['n_inj'] == 0])

    # do the sensitivity calculation using the csky function from trials
    sensitivity = trial_runner.find_n_sig(b.median(), 0.9, tss=trials)
    sens_n_sig = sensitivity['n_sig']
    sens_flux = trial_runner.to_dNdE(sensitivity['n_sig'], E0=1e5, gamma=gamma) / (4 * np.pi * f_sky)
    print('Sensitivity: ', sens_flux)

    # do the discovery potential calculation using the csky function from trials
    #if np.all(df[ts_key] < b.isf_nsigma(5.)):
    #    print('No trials with TS > 5 sigma. Setting discovery potential to 0.')
    #    disc_flux = 0.0
    #    disc_n_sig = 0.0
    #else:
    #    discovery = trial_runner.find_n_sig(b.isf_nsigma(5.), 0.5, tss=trials)
    #    disc_flux = trial_runner.to_dNdE(discovery['n_sig'], E0=1e5, gamma=2.5) / (4 * np.pi * f_sky)
    #    disc_n_sig = discovery['n_sig']
    #print('Discovery potential: ', disc_flux)
    disc_flux = 0.0
    disc_n_sig = 0.0

    output = {
        'sens_flux': sens_flux,
        'sens_n_sig': sens_n_sig,
        'disc_flux': disc_flux,
        'disc_n_sig': disc_n_sig,
        'E0': 1e5,
    }

    return output


if __name__ == '__main__':
    main()