"""Fit a data file that was generated with nuXgal-generate-tomo-data.py"""

import argparse
import json

import numpy as np
from tqdm import tqdm

from KIPAC.nuXgal.TomographicLikelihood import TomographicLikelihood
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

    result_list = []
    for file in args.input:
        json_data = load_json_file(file)
        for data in tqdm(json_data):
            data['galaxy_catalog'] = ['SDSS_z0.0_0.1', 'SDSS_z0.1_0.2', 'SDSS_z0.2_0.3', 'SDSS_z0.3_0.4', 'SDSS_z0.4_0.5', 'SDSS_z0.5_0.6', 'SDSS_z0.6_0.7', 'SDSS_z0.7_0.8', 'SDSS_z0.8_1.0', 'SDSS_z1.0_1.2', 'SDSS_z1.2_1.4', 'SDSS_z1.4_1.6', 'SDSS_z1.6_1.8', 'SDSS_z1.8_2.0', 'SDSS_z2.0_2.2']
            #data['galaxy_catalog'] = data['galaxy_catalog'][:8]
            #data['galaxy_catalog'] = data['galaxy_catalog'][:1]
            data['n_zbins'] = len(data['galaxy_catalog'])
            data['lmin'] = args.lmin
            result_dict = data

            if 'llh' not in locals():
                # temporary fix for the fact that the galaxy catalog is not saved in the json file
                #data['galaxy_catalog'] = ['SDSS_z0.1_0.2', 'SDSS_z0.2_0.3', 'SDSS_z0.3_0.4', 'SDSS_z0.4_0.5']
                llh = TomographicLikelihood.init_from_run(**data, mc_background=True)
                llh.Ncount = np.zeros(Defaults.NEbin)
                for ebin in range(Defaults.NEbin):
                    llh.Ncount[ebin] = data['n_total_i'][str(ebin)]
            else:
                for i in range(data['n_zbins']):
                    llh.llhs[i].w_data = np.zeros((llh.Ebinmax - llh.Ebinmin, Defaults.NCL))
                    for j, ebin in enumerate(range(data['ebinmin'], data['ebinmax'])):
                        llh.llhs[i].w_data[j] = np.array(data['cls'][str(ebin)][i])

                    llh.llhs[i].Ncount = np.array([data['n_total_i'][str(ebin)] for ebin in range(Defaults.NEbin)])


            result_dict = {}

            #f_fit, result_dict["TS_ns_gamma"] = llh.minimize__lnL_ns_gamma()
            f_fit, result_dict["TS_linked"] = llh.minimize__lnL_linked()
            result_dict["params_linked"] = f_fit.tolist()
            result_dict["BIC_linked"] = llh.BIC_linked(*f_fit)
            f_fit, result_dict["TS_SFR"] = llh.minimize__lnL_SFR()
            result_dict["params_SFR"] = f_fit.tolist()
            result_dict["BIC_SFR"] = llh.BIC_SFR(*f_fit)
            f_fit, result_dict["TS_flat"] = llh.minimize__lnL_flat()
            result_dict["params_flat"] = f_fit.tolist()
            result_dict["BIC_flat"] = llh.BIC_flat(*f_fit)
            result_dict["BIC_null"] = llh.BIC_null()
            f_fit, result_dict["TS_cutoff_pl"] = llh.minimize__lnL_cutoff_pl()
            result_dict["params_cutoff_pl"] = f_fit.tolist()
            result_dict["BIC_cutoff_pl"] = llh.BIC_cutoff_pl(*f_fit)
            f_fit, result_dict["TS_independent"] = llh.minimize__lnL_independent()
            result_dict["params_independent"] = f_fit.tolist()
            result_dict["BIC_independent"] = llh.BIC_independent(f_fit[:llh.n_zbins], f_fit[llh.n_zbins:])
            #f_fit, result_dict["TS_ns_gamma"] = llh.minimize__lnL_ns_gamma()
            #result_dict["params_ns_gamma"] = f_fit.tolist()
            #result_dict["BIC_ns_gamma"] = llh.BIC_ns_gamma(f_fit[:-1], f_fit[-1])
            #result_dict["BIC_null"] = llh.BIC_null()
            #result_dict['TS_i_ns_gamma'] = []
            #for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
            #    result_dict['TS_i_ns_gamma'].append(2 * (llh.log_likelihood_cov_Ebin(result_dict['f_fit'][i], ebin) - llh.log_likelihood_cov_Ebin(0, ebin)))
            #result_dict['flux_fit_ns_gamma'] = float(data['n_to_flux'] * result_dict['ns_gamma_fit'][0])
            #result_dict['chi_square_ns_gamma'] = llh.chi_square_ns_gamma(*result_dict['ns_gamma_fit'])

            result_list.append(result_dict)

    with open(args.output, 'w') as f:
        json.dump(result_list, f, indent=4)




if __name__ == "__main__":
    main()
