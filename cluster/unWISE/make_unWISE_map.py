#!/home/dguevel/envs/nuXgal-env/bin/python

from glob import glob
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from astropy.io import fits
import healpy as hp
import astropy.units as u
import astropy
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1
from astroquery.vizier import Vizier


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file name list')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('--nside', default=128)
    parser.add_argument('--cache-dir', default='/data/user/dguevel/astropy/cache', help='astropy cache dir')
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    with open(args.input, 'r') as fp:
        filenames = fp.readlines()

    filenames = [f.strip() for f in filenames]

    nside = args.nside
    with astropy.config.set_temp_cache(args.cache_dir):
        template = read_template_range(filenames, nside)
    np.save(args.output, template)


def read_template_range(files, nside, gaia_filter=False, two_mass_filter=True):
    npix = hp.nside2npix(nside)
    template = np.zeros((4, npix))
    bins = np.arange(npix + 1)

    for filename in files:

        with fits.open(filename) as f:

            unwise = pd.DataFrame()
            unwise['ra'] = f[1].data['ra']
            unwise['dec'] = f[1].data['dec']
            unwise['flux_w1'] = f[1].data['flux'][:,0]
            unwise['flux_w2'] = f[1].data['flux'][:,1]
            unwise['w1'] = 22.5 - 2.5 * np.log10(unwise['flux_w1'])
            unwise['w2'] = 22.5 - 2.5 * np.log10(unwise['flux_w2'])
            unwise['primary'] = f[1].data['primary']

            #idx = (f[1].data['primary'] == 1) * (f[1].data['flux'][:,0] > 0) * (f[1].data['flux'][:,1] > 0)
            #idx *= (f[1].data['flags_unwise'][:,0] == 0) * (f[1].data['flags_unwise'][:,1] == 0)

            ra = f[0].header['CRVAL1']
            dec = f[0].header['CRVAL2']
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            gal_b = coord.transform_to('galactic').b
            unwise_coord = SkyCoord(unwise['ra'], unwise['dec'], unit=u.deg)

            # skip areas we aren't using
            if (np.abs(gal_b) > 5 * u.deg) and (coord.dec > -5 * u.degree):
                if gaia_filter:
                    pixel_scale = (np.abs(f[0].header['CD1_1']) + np.abs(f[0].header['CD2_2'])) / 2.
                    search_radius = max(f[0].header['IMAGEW'], f[0].header['IMAGEH']) * pixel_scale * u.deg

                    query = Gaia.cone_search_async(coord, radius=search_radius)
                    gaia_cat = query.get_results()
                    gaia_coord = SkyCoord(gaia_cat['ra'], gaia_cat['dec'])

                    unwise['gaia_sep'] = unwise_coord.match_to_catalog_sky(gaia_coord)[1].to(u.arcsec).value
                    
                    #idx *= (sep2d > 5*u.arcsec)


                if two_mass_filter:
                    # pixel scale is defined in the transformation matrix from pixel to sky coordinate
                    # see WCS standards for details
                    pixel_scale = (np.abs(f[0].header['CD1_1']) + np.abs(f[0].header['CD2_2'])) / 2.
                    search_radius = max(f[0].header['IMAGEW'], f[0].header['IMAGEH']) * pixel_scale * u.deg

                    viz_2mass = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag'], catalog='II/246', column_filters={}, row_limit=-1)
                    result = viz_2mass.query_region(coord, radius=search_radius)
                    if len(result) > 0:
                        two_mass = result[0]
                        two_mass_coord = SkyCoord(ra=two_mass['RAJ2000'], dec=two_mass['DEJ2000'], unit=u.deg)
                        two_mass_xmatch = unwise_coord.match_to_catalog_sky(two_mass_coord)
                        unwise['2mass_sep'] = two_mass_xmatch[1].to(u.arcsec).value
                        unwise['J'] = two_mass[two_mass_xmatch[0]]['Jmag']
                        #idx *= (two_mass_xmatch[1] < 3 * u.arcsec) * (two_mass[two_mass_xmatch[0]]['Jmag'] < 16.5)
                    else:
                        unwise['2mass_sep'] = 1e9
                        unwise['J'] = 25

                unwise['pixel'] = hp.ang2pix(nside, unwise['ra'].array, unwise['dec'].array, lonlat=True)
                idx = (unwise['primary'] == 1) * (unwise['flux_w1'] > 0) * (unwise['flux_w2'] > 0)
                if two_mass_filter:
                    idx &= (unwise['J'] < 16.5) & (unwise['w1'] - unwise['J'] < -1.7) & (unwise['2mass_sep'] <= 3)


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
                        
                    template[i] += np.histogram(unwise['pixel'][idx & zidx], bins=bins)[0]
    return template

if __name__ == '__main__':
    main()