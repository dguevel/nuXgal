#!/home/dguevel/envs/nuXgal-env/bin/python

"""Make unWISE template maps including the cross matching
unWISE and 2MASS catalogs. The following color and magnitude 
cuts are also applied: W1 - J < -1.7, J < 16.5.

See https://doi.org/10.1093/mnras/stv063 and http://arxiv.org/abs/1901.03337
"""

import os
from argparse import ArgumentParser
import json


import numpy as np
import pandas as pd
from astropy.io import fits
import healpy as hp
from tqdm import tqdm
import astropy.units as u
import astropy
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Input unWISE file name list')
    parser.add_argument('-o', '--output', type=str,
                        help='Output galaxy map file name save as np array')
    parser.add_argument('--nside',
                        help='Healpix nside parameter for output map',
                        default=128)
    parser.add_argument('--cache-dir',
                        default='/data/user/dguevel/astropy/cache',
                        help='astropy cache dir')
    parser.add_argument('--logfile', type=str, help='Filename for log file',
                        default='unWISE_analytics.json')
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    with open(args.input, 'r') as fp:
        filenames = fp.readlines()

    filenames = [f.strip() for f in filenames]

    nside = args.nside
    with astropy.config.set_temp_cache(args.cache_dir):
        template, info_list = read_template_range(filenames, nside)
    np.save(args.output, template)
    with open(args.logfile, 'w') as fp:
        json.dump(info_list, fp, indent=4)


def read_template_range(files, nside, cross_match_radius=3):
    """
    Read a list of files and return a galaxy map in healpix format

    Parameters:
        files (list): List of file names.
        nside (int): The nside parameter for healpy.
        cross_match_radius (float): Cross match radius in arcsec.

    Returns:
        numpy.ndarray: The galaxy map as a healpy and numpy array.
    """
    npix = hp.nside2npix(nside)
    template = np.zeros((4, npix))
    bins = np.arange(npix + 1)

    info_list = []

    for filename in tqdm(files):

        info_dict = {}
        info_dict['filename'] = filename

        with fits.open(filename) as f:

            # read basic quantities from unWISE catalog
            unwise = pd.DataFrame()
            unwise['ra'] = f[1].data['ra'].byteswap().newbyteorder()
            unwise['dec'] = f[1].data['dec'].byteswap().newbyteorder()
            # put in a minimum flux to avoid log10(0)
            unwise['flux_w1'] = np.maximum(f[1].data['flux'][:, 0].byteswap().newbyteorder(), 1e-30)
            unwise['flux_w2'] = np.maximum(f[1].data['flux'][:, 1].byteswap().newbyteorder(), 1e-30)
            unwise['w1'] = 22.5 - 2.5 * np.log10(unwise['flux_w1'])
            unwise['w2'] = 22.5 - 2.5 * np.log10(unwise['flux_w2'])
            unwise['primary'] = f[1].data['primary'].byteswap().newbyteorder()
            info_dict['total_unwise_objects'] = len(unwise)
            # data quality cut
            dq_idx = unwise['primary'] == 1
            info_dict['good_unwise_objects'] = np.sum(dq_idx).item()

            # load ra, dec of the center of the image using WCS
            ra = f[0].header['CRVAL1']
            dec = f[0].header['CRVAL2']
            coord = SkyCoord(ra=ra, dec=dec, unit=u.degree, frame='icrs')
            gal_coord = coord.transform_to('galactic')
            unwise_coord = SkyCoord(unwise['ra'], unwise['dec'], unit=u.deg)

            info_dict['center_ra'] = ra
            info_dict['center_dec'] = dec
            info_dict['center_l'] = gal_coord.l.value.item()
            info_dict['center_b'] = gal_coord.b.value.item()

            # pixel scale is defined in the transformation matrix from
            # pixel to sky coordinate see WCS standards for details
            pixel_scale = (np.abs(f[0].header['CD1_1'])
                           + np.abs(f[0].header['CD2_2'])) / 2.
            search_radius = max(f[0].header['IMAGEW'], f[0].header['IMAGEH'])
            search_radius *= pixel_scale * u.deg

            info_dict['2mass_search_radius'] = search_radius.to(u.arcsec).value.item()

            # query 2MASS catalog from Vizier
            # TODO: set up local copy of the catalog
            viz_2mass = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag'],
                               catalog='II/246',
                               column_filters={},
                               row_limit=-1)
            result = viz_2mass.query_region(coord, radius=search_radius)
            if len(result) > 0:
                two_mass = result[0]
                two_mass_coord = SkyCoord(ra=two_mass['RAJ2000'],
                                          dec=two_mass['DEJ2000'], unit=u.deg)
                two_mass_xmatch = unwise_coord.match_to_catalog_sky(two_mass_coord)
                unwise['2mass_sep'] = two_mass_xmatch[1].to(u.arcsec).value
                unwise['J'] = two_mass[two_mass_xmatch[0]]['Jmag']
            else:
                # arbitrary values that won't pass the color and magnitude cuts
                unwise['2mass_sep'] = 1000
                unwise['J'] = 1000

            info_dict['total_2mass_objects'] = len(two_mass)
            xmatch_idx = (unwise['2mass_sep'] <= cross_match_radius)
            info_dict['2mass_unwise_matches'] = (xmatch_idx & dq_idx).sum().item()

            # get pixel index for each object
            unwise['pixel'] = hp.ang2pix(nside, unwise['ra'].array,
                                         unwise['dec'].array, lonlat=True)

            # color and magnitude cut
            color_idx = (unwise['J'] < 16.5)
            color_idx &= (unwise['w1'] - unwise['J'] < -1.7)

            # apply color and magnitude cuts for redshift selection
            info_dict['2mass_unwise_z_bin_total'] = []
            info_dict['2mass_unwise_z_bin_xmatch'] = []
            info_dict['2mass_unwise_z_bin_filtered'] = []
            for i in range(4):
                if i == 0:
                    zidx = unwise['w2'] < 15.5
                elif i == 1:
                    zidx = unwise['w2'] > 15.5
                    zidx *= (unwise['w1'] - unwise['w2']) < ((17-unwise['w2'])/4. + 0.3)
                elif i == 2:
                    zidx = unwise['w2'] > 15.5
                    zidx *= (unwise['w1'] - unwise['w2']) > ((17-unwise['w2'])/4. + 0.3)
                    zidx *= (unwise['w1'] - unwise['w2']) < ((17-unwise['w2'])/4. + 0.8)
                elif i == 3:
                    zidx = unwise['w2'] > 15.5
                    zidx *= (unwise['w1'] - unwise['w2']) > ((17-unwise['w2'])/4. + 0.8)

                idx = color_idx & zidx & xmatch_idx & dq_idx
                template[i] += np.histogram(unwise['pixel'][idx], bins=bins)[0]

                info_dict['2mass_unwise_z_bin_total'].append((zidx & dq_idx).sum().item())
                info_dict['2mass_unwise_z_bin_xmatch'].append((zidx & xmatch_idx & dq_idx).sum().item())
                info_dict['2mass_unwise_z_bin_filtered'].append((idx).sum().item())

            info_list.append(info_dict)


    return template, info_list
            

if __name__ == '__main__':
    main()
