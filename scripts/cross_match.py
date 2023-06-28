
from argparse import ArgumentParser

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', type=int, help='Healpix pixel index')
    parser.add_argument('--separation', default=5., help='Catalog cross match separation threshold in arcsec')
    parser.add_argument('--nside', default=128)
    args = parser.parse_args()

    ra, dec = hp.pix2ang(args.nside, args.input, lonlat=True)
    search_rad = 2 * hp.nside2resol(args.nside) * 180 / np.pi
    n_galaxies = np.zeros(len(args.input))
    n_total = np.zeros(len(args.input))
    for i, (r, d) in enumerate(zip(ra, dec)):
        coord = SkyCoord(ra=r, dec=d, unit=(u.degree, u.degree), frame='icrs')
        print(coord)

        j = Gaia.cone_search_async(coord, radius=search_rad*u.deg)
        gaia_cat = j.get_results()
        gaia_coord = SkyCoord(gaia_cat['ra'], gaia_cat['dec'])

        k = Vizier.query_region(coord, radius=search_rad*u.deg, catalog='II/363/unwise')
        unwise_cat = k[0]
        unwise_coord = SkyCoord(ra=unwise_cat['RAJ2000'], dec=unwise_cat['DEJ2000'], unit=u.deg)

        idx, sep2d, sep3d = unwise_coord.match_to_catalog_sky(gaia_coord)

        good_galaxies = (unwise_cat['q_W1'] == 1) & (unwise_cat['q_W2'] == 1) & (unwise_cat['FW1'] > 0) & (unwise_cat['FW2'] > 0) & (sep2d > args.separation*u.arcsec)
        all_galaxies = (unwise_cat['q_W1'] == 1) & (unwise_cat['q_W2'] == 1) & (unwise_cat['FW1'] > 0) & (unwise_cat['FW2'] > 0)# & (sep2d > args.separation*u.arcsec)


        w1 = 22.5 - 2.5 * np.log10(unwise_cat['FW1'][good_galaxies])
        w2 = 22.5 - 2.5 * np.log10(unwise_cat['FW1'][good_galaxies])

        n_galaxies[i] = (w1 < 15.5).sum()
        w2 = 22.5 - 2.5 * np.log10(unwise_cat['FW1'][all_galaxies])
        n_total[i] = (w2 < 15.5).sum()
        print(n_galaxies[i], n_total[i])

    print(n_galaxies)
    print(n_total)
    print(n_galaxies/n_total)

if __name__ == '__main__':
    main()