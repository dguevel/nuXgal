"""Fit a data file that was generated with nuXgal-generate-data.py"""

import argparse
import json

import numpy as np
from tqdm import tqdm

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal import Defaults


def load_json_file(file):
    with open(file, 'r') as f:
        json_data = json.load(f)
    return json_data


def main():
    parser = argparse.ArgumentParser(description="Load JSON files")
    parser.add_argument("-i", "--input", nargs="+", help="JSON files to load")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("--lmin", default=1, type=int, help="Minimum multipole to use")
    parser.add_argument("--gamma", default=2.5, type=float, help="Fit model spectral index")
    parser.add_argument("--path-sig", default='', help='Path to MC data set for fitting')

    args = parser.parse_args()

    result_list = []
    for file in args.input:
        json_data = load_json_file(file)
        for data in tqdm(json_data):
            result_dict = data
            data['lmin'] = args.lmin
            data['gamma'] = args.gamma
            data['mc_background'] = True
            data['path_sig'] = args.path_sig
            data['galaxy_catalog'] = 'unWISE_z=0.4'
            if 'llh' not in locals():
                llh = Likelihood.init_from_run(**data)
                llh.Ncount = np.zeros(Defaults.NEbin)
                for ebin in range(Defaults.NEbin):
                    llh.Ncount[ebin] = data['n_total_i'][str(ebin)]
                    llh.Ncount_unweighted = np.array([data['n_total_i'][str(ebin)] for ebin in range(Defaults.NEbin)])
            else:
                llh.w_data = np.zeros((llh.Ebinmax - llh.Ebinmin, Defaults.NCL))
                for j, ebin in enumerate(range(data['ebinmin'], data['ebinmax'])):
                    llh.w_data[j] = np.array(data['cls'][str(ebin)])

                llh.Ncount = np.array([data['n_total_i'][str(ebin)] for ebin in range(Defaults.NEbin)])
                llh.Ncount_unweighted = np.array([data['n_total_i'][str(ebin)] for ebin in range(Defaults.NEbin)])

            f_fit, result_dict["TS"] = llh.minimize__lnL_free_atm()
            result_dict["f_fit"] = f_fit.tolist()
            result_dict['TS_i'] = []
            result_dict['chi_square'] = []
            result_dict['n_fit'] = []
            for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
                result_dict['TS_i'].append(2 * (llh.log_likelihood_cov_Ebin(result_dict['f_fit'][i], ebin) - llh.log_likelihood_cov_Ebin(0, ebin)))
                result_dict['chi_square'].append(llh.chi_square_cov_Ebin(result_dict['f_fit'][i], ebin))
                result_dict['n_fit'].append(result_dict['f_fit'][i] * data['n_total_i'][str(ebin)])
            result_dict['flux_fit'] = float(data['n_to_flux'] * sum(result_dict['n_fit']))

            f_fit, result_dict["TS_f_gamma"] = llh.minimize__lnL_free_bg()
            result_dict["f_gamma_fit"] = f_fit.tolist()
            result_dict["chi_square_f_gamma"] = llh.chi_square_free_bg(f_fit)

            result_list.append(result_dict)

    with open(args.output, 'w') as f:
        json.dump(result_list, f, indent=4)




if __name__ == "__main__":
    main()
