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
    args = parser.parse_args()

    cov = np.load('/home/dguevel/git/nuXgal/syntheticData/nt_v5_mckde.npy')
    std = np.array([np.sqrt(np.diag(cov[i])) for i in range(cov.shape[0])])
    #w_model_f1 = np.load('/home/dguevel/git/nuXgal/syntheticData/nt_v5_many_iter_w_mean.npy')
    #w_atm_mean = np.load('/home/dguevel/git/nuXgal/syntheticData/nt_v5_mcbg_w_mean.npy')

    result_list = []
    _event_generators = None
    for file in args.input:
        json_data = load_json_file(file)
        for data in tqdm(json_data):
            result_dict = data
            llh = Likelihood.init_from_run(**data, lmin=args.lmin)
            llh.Ncount = np.zeros(Defaults.NEbin)
            llh.w_atm_mean *= 0
            for ebin in range(Defaults.NEbin):
                llh.Ncount[ebin] = data['n_total_i'][str(ebin)]
            #if _event_generators:
            #    llh._event_generators = _event_generators
            #else:
            #    llh._get_event_generators()
            #    _event_generators = llh._event_generators

            llh.w_std = std
            llh.w_cov = cov
            #result_dict['TS_diag'] = llh.minimize__lnL()[1]
            f_fit, result_dict["TS"] = llh.minimize__lnL_cov()
            result_dict["f_fit"] = f_fit.tolist()
            result_dict['TS_i'] = []
            result_dict['chi_square'] = []
            result_dict['n_fit'] = []
            for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
                result_dict['TS_i'].append(2 * (llh.log_likelihood_cov_Ebin(result_dict['f_fit'][i], ebin) - llh.log_likelihood_cov_Ebin(0, ebin)))
                result_dict['chi_square'].append(llh.chi_square_cov_Ebin(result_dict['f_fit'][i], ebin))
                result_dict['n_fit'].append(result_dict['f_fit'][i] * data['n_total_i'][str(ebin)])
            result_dict['flux_fit'] = float(data['n_to_flux'] * sum(result_dict['n_fit']))

            #f_fit, result_dict["TS_ns_gamma"] = llh.minimize__lnL_ns_gamma()
            #result_dict["ns_gamma_fit"] = f_fit.tolist()
            #result_dict['TS_i_ns_gamma'] = []
            #for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
            #    result_dict['TS_i_ns_gamma'].append(2 * (llh.log_likelihood_cov_Ebin(result_dict['f_fit'][i], ebin) - llh.log_likelihood_cov_Ebin(0, ebin)))
            #result_dict['flux_fit_ns_gamma'] = float(data['n_to_flux'] * result_dict['ns_gamma_fit'][0])
            
            result_list.append(result_dict)

    with open(args.output, 'w') as f:
        json.dump(result_list, f, indent=4)




if __name__ == "__main__":
    main()
