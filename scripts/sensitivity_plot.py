import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from KIPAC.nuXgal import Defaults

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='Input files from sensitivity.py')
    parser.add_argument('-o', '--output', help='Output file', default='sensitivity-plot.pdf')
    parser.add_argument('--no-template', action='store_true', help='Do not plot template sensitivity')

    args = parser.parse_args()

    # load data
    print('Reading files...')
    data = []
    for fname in args.input:
        with open(fname) as f:
            print(f'Reading {fname}')
            data.append(json.load(f))

    plot_ns_sensitivity(data, args)
    plot_ereco_sensitivity(data, args)

def plot_ns_sensitivity(data, args):

    xunit = u.GeV

    fig, ax = plt.subplots(dpi=150)

    cc_sens = np.array([d['sens_n_sig'] for d in data])
    cc_sens_yerr = np.array([d['sens_n_sig_err'] for d in data])
    tmp_sens = np.array([d['template_sens_n_sig'] for d in data])
    tmp_sens_yerr = np.array([d['template_sens_n_sig_err'] for d in data])
    cc_disc = np.array([d['disc_n_sig'] for d in data])
    cc_disc_yerr = np.array([d['disc_n_sig_err'] for d in data])
    tmp_disc = np.array([d['template_disc_n_sig'] for d in data])
    tmp_disc_yerr = np.array([d['template_disc_n_sig_err'] for d in data])
    munu_spl = np.array([d['munu_ns_spl'] for d in data])
    munu_spl_err = np.array([d['munu_ns_spl_std'] for d in data])

    xmid = []
    xlo = []
    xhi = []
    for d in data:
        xmid.append(d['E0'])
        xlo.append(xmid[-1] - 10**d['logemin'])
        xhi.append(10**d['logemax'] - xmid[-1])

    xmid = np.array(xmid) * u.GeV
    xerr = np.array([xlo, xhi]) * u.GeV

    ax.errorbar(xmid.to(xunit).value, cc_sens, xerr=xerr.to(xunit).value, yerr=cc_sens_yerr, fmt='x', color='black', markersize=8, label='Cross Correlation Sensitivity')
    if not args.no_template:
        ax.errorbar(xmid.to(xunit).value, tmp_sens, xerr=xerr.to(xunit).value, yerr=tmp_sens_yerr, fmt='o', color='C0', markersize=5, label='Template Sensitivity')

    xmid = []
    xlo = []
    xhi = []
    for d in data:
        xmid.append(d['E0'])
        xlo.append(xmid[-1] - 10**d['logemin'])
        xhi.append(10**d['logemax'] - xmid[-1])

    xmid = np.array(xmid) * u.GeV
    xerr = np.array([xlo, xhi]) * u.GeV

    eb = ax.errorbar(xmid.to(xunit).value, cc_disc, xerr=xerr.to(xunit).value, yerr=cc_disc_yerr, fmt='x', color='black', markersize=8, label='Cross Correlation Discovery Potential')
    eb[-1][0].set_linestyle('--')
    if not args.no_template:
        eb = ax.errorbar(xmid.to(xunit).value, tmp_disc, xerr=xerr.to(xunit).value, yerr=tmp_disc_yerr, fmt='o', color='C0', markersize=5, label='Template Discovery Potential')
        eb[-1][0].set_linestyle('--')

    ax.errorbar(xmid, munu_spl, yerr=munu_spl_err, fmt='o', color='C1', markersize=5, label=r'Diffuse $\nu_\mu$ power law')

    ax.loglog()
    ax.set_ylim(8, 20000)
    ax.legend(loc='lower left')
    ax.set_xlabel(r'$E_{reco}$ / ' + xunit.to_string(format='latex_inline'))
    ax.set_ylabel(r'Number of $\nu_\mu$')
    ax.set_title('Northern Tracks v5 Sensitivity')

    plt.savefig(args.output.replace('.', '_ns.'), bbox_inches='tight')



def plot_ereco_sensitivity(data, args):

    xunit = u.GeV
    yunit = u.GeV/u.cm**2/u.s/u.sr

    fig, ax = plt.subplots(dpi=150)

    # plot cross correlation sensitivity
    cc_sens = np.array([d['sens_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    cc_sens_yerr = np.array([d['sens_flux_err'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    tmp_sens = np.array([d['template_sens_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    tmp_sens_err = np.array([d['template_sens_flux_err'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    xmid = []
    xlo = []
    xhi = []
    for d in data:
        xmid.append(d['E0'])
        xlo.append(xmid[-1] - 10**d['logemin'])
        xhi.append(10**d['logemax'] - xmid[-1])

    xmid = np.array(xmid) * u.GeV
    xerr = np.array([xlo, xhi]) * u.GeV
    cc_sens = xmid**2 * cc_sens
    tmp_sens = xmid**2 * tmp_sens
    cc_sens_yerr = xmid**2 * cc_sens_yerr
    tmp_sens_yerr = xmid**2 * tmp_sens_err

    ax.errorbar(xmid.to(xunit).value, cc_sens.to(yunit).value, xerr=xerr.to(xunit).value, yerr=cc_sens_yerr.to(yunit).value, fmt='x', color='black', markersize=8, label='Cross Correlation Sensitivity')
    if not args.no_template:
        ax.errorbar(xmid.to(xunit).value, tmp_sens.to(yunit).value, xerr=xerr.to(xunit).value, yerr=tmp_sens_yerr.to(yunit).value, fmt='o', color='C0', markersize=5, label='Template Sensitivity')

    # plot cross correlation discovery potential
    cc_disc = np.array([d['disc_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    tmp_disc = np.array([d['template_disc_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    cc_disc_yerr = np.array([d['disc_flux_err'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    tmp_disc_yerr = np.array([d['template_disc_flux_err'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    print(cc_sens, tmp_sens)
    xmid = []
    xlo = []
    xhi = []
    for d in data:
        xmid.append(d['E0'])
        xlo.append(xmid[-1] - 10**d['logemin'])
        xhi.append(10**d['logemax'] - xmid[-1])

    xmid = np.array(xmid) * u.GeV
    xerr = np.array([xlo, xhi]) * u.GeV
    cc_disc = xmid**2 * cc_disc# * ((xmid / 100 / u.TeV).si)**(-2.5)
    tmp_disc = xmid**2 * tmp_disc# * ((xmid / 100 / u.TeV).si)**(-2.5)
    cc_disc_yerr = xmid**2 * cc_disc_yerr
    tmp_disc_yerr = xmid**2 * tmp_disc_yerr

    eb = ax.errorbar(xmid.to(xunit).value, cc_disc.to(yunit).value, xerr=xerr.to(xunit).value, yerr=cc_disc_yerr.to(yunit).value, fmt='x', color='black', markersize=8, label='Cross Correlation Discovery Potential')
    eb[-1][0].set_linestyle('--')
    if not args.no_template:
        eb = ax.errorbar(xmid.to(xunit).value, tmp_disc.to(yunit).value, xerr=xerr.to(xunit).value, yerr=tmp_disc_yerr.to(yunit).value, fmt='o', color='C0', markersize=5, label='Template Discovery Potential')
        eb[-1][0].set_linestyle('--')

    # single power law
    energy = np.linspace(15, 2000, 100) * u.TeV
    alpha = -2.37
    E0 = 100 * u.TeV
    phi_err=0.25e-18 /u.GeV/u.cm**2/u.s/u.sr
    alpha_err = 0.09
    dNdE0 = 1.44e-18 /u.GeV/u.cm**2/u.s/u.sr
    dNdE = dNdE0 * ((energy/E0).si ** alpha)
    dNdE_err = np.sqrt(dNdE**2/dNdE0**2*phi_err**2 + dNdE0**2*(energy/E0)**(2*alpha)*np.log((energy/E0).si.value)**2*alpha_err**2)
    E2dNdE_err = (energy**2 * dNdE_err).to(yunit)
    E2dNdE = (energy**2 * dNdE).to(yunit)

    ax.plot(energy.to(xunit).value, E2dNdE.value, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(energy.to(xunit).value, E2dNdE.value-E2dNdE_err.value, E2dNdE.value+E2dNdE_err.value, alpha=0.5, label=r'Diffuse $\nu_\mu$ flux (Abbasi et al 2021)')

    ax.loglog()
    ax.legend()
    ax.set_ylim(2e-9, 4e-6)
    ax.set_xlabel(r'$E_{reco}$ / ' + xunit.to_string(format='latex_inline'))
    ax.set_ylabel(r'$E^2 \frac{dN}{dE}$ / ' + yunit.to_string(format='latex_inline'))
    ax.set_title('Northern Tracks v5 Sensitivity')

    plt.savefig(args.output, bbox_inches='tight')

if __name__ == '__main__':
    main()
