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

    xunit = u.GeV
    yunit = u.GeV/u.cm**2/u.s/u.sr

    # load data
    print('Reading files...')
    data = []
    for fname in args.input:
        with open(fname) as f:
            print(f'Reading {fname}')
            data.append(json.load(f))

    fig, ax = plt.subplots()

    # plot cross correlation sensitivity
    cc_sens = np.array([d['sens_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    tmp_sens = np.array([d['template_sens_flux'] for d in data]) / u.GeV / u.cm**2 / u.s / u.sr
    xmid = []
    xlo = []
    xhi = []
    for d in data:
        xmid.append(d['E0'])
        xlo.append(xmid[-1] - 10**d['logemin'])
        xhi.append(10**d['logemax'] - xmid[-1])

    xmid = np.array(xmid) * u.GeV
    xerr = np.array([xlo, xhi]) * u.GeV
    cc_sens = xmid**2 * cc_sens# * ((xmid / 100 / u.TeV).si)**(-2.5)
    tmp_sens = xmid**2 * tmp_sens# * ((xmid / 100 / u.TeV).si)**(-2.5)
    cc_sens_yerr = 0.2 * cc_sens.to(yunit).value
    tmp_sens_yerr = 0.2 * tmp_sens.to(yunit).value

    ax.errorbar(xmid.to(xunit).value, cc_sens.to(yunit).value, xerr=xerr.to(xunit).value, yerr=cc_sens_yerr, fmt='x', color='black', uplims=True, markersize=8, label='Cross Correlation Sensitivity')
    if not args.no_template:
        ax.errorbar(xmid.to(xunit).value, tmp_sens.to(yunit).value, xerr=xerr.to(xunit).value, yerr=tmp_sens_yerr, fmt='o', color='C0', uplims=True, markersize=5, label='Template Sensitivity')

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
    ax.set_xlabel(r'$E_\nu$ / ' + xunit.to_string(format='latex_inline'))
    ax.set_ylabel(r'$E^2 \frac{dN}{dE}$ / ' + yunit.to_string(format='latex_inline'))
    ax.set_title('Sensitivity')

    plt.savefig(args.output, bbox_inches='tight')

if __name__ == '__main__':
    main()