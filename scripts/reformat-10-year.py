"""
Script to reformat the IceCube 10 year point source data effective area files
to match the 3 year point source data files.
"""

import argparse
import os
import re

import numpy as np

from KIPAC.nuXgal import Defaults

def load_aeff_file(aeff_fname):
    """
    Read an effective area file.

    Input
    -----
    aeff_fname : str
        Name of effective area file.

    Return
    ------
    columns : numpy record array
        Columns of effective area file.
    """

    return np.genfromtxt(aeff_fname, names=True, deletechars='')

def reformat_aeff_table(aeff_table):
    """
    Reformat an effective area table.

    Modify the IceCube 10-year data release effective area files to match the
    format of the 3-year data release effective area files.

    1. Reorder the flattening of the aeff data cube.
    2. Fix units.
    3. Convert declination to zenith angle.

    Input
    -----
    aeff_table : numpy record array
        Aeff table from load_aeff_file

    Return
    ------
    new_aeff_table : numpy record array
    """

    # fix units
    col_names = list(aeff_table.dtype.names)

    aeff_table['A_Eff[cm^2]'] *= 1e-4
    col_names[col_names.index('A_Eff[cm^2]')] = 'A_Eff[m^2]'

    aeff_table['log10(E_nu/GeV)_min'] = np.power(10, aeff_table['log10(E_nu/GeV)_min'])
    col_names[col_names.index('log10(E_nu/GeV)_min')] = 'E_min[GeV]'

    aeff_table['log10(E_nu/GeV)_max'] = np.power(10, aeff_table['log10(E_nu/GeV)_max'])
    col_names[col_names.index('log10(E_nu/GeV)_max')] = 'E_max[GeV]'

    aeff_table['Dec_nu_min[deg]'] = np.cos(np.radians(90 - aeff_table['Dec_nu_min[deg]']))
    col_names[col_names.index('Dec_nu_min[deg]')] = 'cos(zenith)_min'

    aeff_table['Dec_nu_max[deg]'] = np.cos(np.radians(90 - aeff_table['Dec_nu_max[deg]']))
    col_names[col_names.index('Dec_nu_max[deg]')] = 'cos(zenith)_max'

    aeff_table.dtype.names = tuple(col_names)

    # reorder the table
    new_aeff_table = aeff_table.copy()
    n = np.unique(aeff_table['E_min[GeV]']).size
    m = np.unique(aeff_table['cos(zenith)_max']).size
    for i in np.arange(n):
        new_aeff_table[i*m:(i+1)*m] = aeff_table[i::n]

    return new_aeff_table



def write_aeff_table(aeff_table, output_filename):
    """
    Write an effective area table.

    Inputs
    ------
    aeff_table : numpy record array
        Aeff table data
    output_filename : str
        Output file name including directory
    """

    header = '\t'.join(aeff_table.dtype.names)
    np.savetxt(output_filename, aeff_table, header=header, fmt='\t%2.3e\t%2.3e\t%1.2f\t%1.2f\t%2.3e')

def main():
    parser = argparse.ArgumentParser('Convert IceCube 10 year point source data\
effective area files to match the format of the 3 year data')
    parser.add_argument('-i', '--input', required=True, help='Input file directory')
    parser.add_argument('-o', '--output', default=Defaults.NUXGAL_IRF_DIR, help='Output file directory')
    args = parser.parse_args()

    # iterate over files in the input directory
    for fname in os.listdir(args.input):
        # check if file is effective area file and capture year
        match = re.match('^(?P<year>IC[0-9]{2}[_IV]{0,4})_effectiveArea.csv$', fname)
        if match:

            fullpath = os.path.join(args.input, fname)
            aeff_data = load_aeff_file(fullpath)
            aeff_data = reformat_aeff_table(aeff_data)
            
            output_filename = Defaults.TABULATED_AEFF_FORMAT.format(year=match.group('year'))
            write_aeff_table(aeff_data, os.path.join(args.output, output_filename))


if __name__ == '__main__':
    main()
