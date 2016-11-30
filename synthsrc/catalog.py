# PYTHON_ARGCOMPLETE_OK
import os
from glob import glob
import argcomplete, argparse
import numpy as np
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt

from templates import *
from lfs import LuminosityFunction
from source import Source
from utils import Cosmology


class Catalog(object):
    """Class to build a catalog of synthetic photometry.

    some stuff

    Two directories will be created in :attr:`outdir`: ``outputs``,
    which will contain the output catalog and logfile, and ``plots``,
    which will contain plots of the redshift distribution, color-color
    diagram, and selection window diagnostics. 

    Args:
        nsrc (int): Number of synthetic sources to create
        add_lines (bool): If True, emission lines will be added to the spectra 
            of the sources.             
        outdir (Optional[str]): Directory for simulation outputs (catalogs
            and plots). Defaults to current working directory.
    """

    def __init__(self, nsrc, add_lines, outfits, outdat, logfile):
        """Initializes the Catalog class."""
        self.nsrc = nsrc
        self.add_lines = add_lines

        self.outfits = outfits
        self.outdat = outdat
        self.logfile = logfile

        # initialize cosmology
        self.cosmo = Cosmology(h=0.7, omega_m=0.3, omega_l=0.7, 
                               log=self.logfile)

        self.get_luminosity_functions()
        self.zz = self.generate_redshifts(nsrc, lowz=0.1, plotdist=True)

        # wavelength array for interpolation of filters and SED templates 
        self.wave_array = np.arange(1, 45000.)

        # read in filters
        self.filters = Filter(self.wave_array)

        # read in templates
        self.temp = Template()
        
        self.build_catalog()


    def get_luminosity_functions(self):
        """Creates instances of the LFs appropriate for each redshift range.

        The luminosity of each source is pulled randomly from the 
        appropriate luminosity function. In each redshift range, 
        the cumulative LF is calculated by integrating the LF from 
        :math:`10L^*` down to the luminosity corresponding to the 
        :math:`1\sigma` magnitude limit of our survey. 
        The magnitude limits are the median :math:`1\sigma` limits in
        the filters covering the LFs' characteristic wavelengths
        in the redshift ranges. 
        """
        # median maglims
        maglims = {'0':[27.672,8024], '1':[27.672,8024], '2':[28.135,5887], \
                   '3':[28.135,5887], '4':[27.672,8024], '5':[28.320,11534], \
                   '6':[28.320,11534], '7':[28.320,11534], '8':[27.422,15369]}
        # max maglims
#        maglims = {'1':[27.971,8024], '2':[28.488,5887], '3':[28.488,5887], \
#                   '4':[27.971,8024], '5':[28.693,11534], '6':[28.693,11534], \
#                   '7':[28.693,11534], '8':[27.696,15369]}

        for k,v in maglims.iteritems():
            setattr(self, 'LF%s'%k, LuminosityFunction(k,v, self.cosmo, 
                                                       log=self.logfile))


    def generate_redshifts(self, nsrc, lowz=0.1, highz=8.5, plotdist=True):
        """Creates a flat redshift distribution of sources in range lowz < z < highz.
    
        Redshifts are pulled from a uniform distribution using 
        :func:`numpy.random.uniform`.

        Args:
            nsrc (int): Number of sources 
            lowz (Optional[float]): Lower redshift of distribution.
                Defaults to 0.1
            highz (Optional[float]): Upper redshift of distribution.
                Defaults to 8.5
            plotdist (Optional[bool]): If true, plots the generated
                distribution as a histogram and saves to ``plots/zdist.pdf``.

        Returns:
            zz (float): Array of randomly generated redshifts.

        """
        #bins = np.linspace(lowz, highz, nbins+1)
        ## add a few extra sources if necessary
        #if nsrc % nbins != 0:
        #    nsrc += nbins - (nsrc % nbins)
        ## sources per bin
        #ns = nsrc / nbins
        #zz = np.zeros(nsrc, dtype=float)
        #for i in range(nbins):
        #    j = i * ns
        #    zz[j:j+ns] = np.random.uniform(bins[i], bins[i+1], ns)
        zz = np.random.uniform(lowz, highz, nsrc)

        log = open(self.logfile, 'a')
        log.write('\n%i redshifts generated over %.1f < z < %.1f\n' % 
                  (nsrc,lowz,highz))    
        log.close()

        if plotdist:
            n,b = np.histogram(zz, bins=100)
            bc = 0.5*(b[1:] + b[:-1])
            w = b[1] - b[0]
            plt.bar(bc, n, align='center', width=w, linewidth=0, alpha=0.5)
            plt.xlabel(r'$z$', fontsize=15)
            plt.ylabel(r'$n_{\mathrm{sources}}$', fontsize=15)
            plt.tight_layout()
            plt.savefig('plots/zdist.pdf')

        return zz


    def build_catalog(self):
        """Builds the catalog of synthetic photometry.

        
        All physical parameters are assigned by the :class:`Source` class.
        :func:`Filter.calc_synphot` converts the synthetic spectrum into 
        flux densities in each filter. These in turn are converted into 
        AB magnitudes. The output is output in both FITS and ascii 
        formats in the ``outputs`` directory.
        """

        names = ['z', 'zLF', 'Mst', 'Lst', 'lum', 'bc03', 'Uage', 'age', \
                 'em', 'nly', 'extlaw', 'nclumps', 'ebv', 'tauv_clump', \
                 'm606', 'm814', 'm110', 'm160', 'irac_ch1']
        dtype = [float, int, float, float, float, 'S30', float, float, \
                 int, float, 'S10', int, float, float, float, float, \
                 float, float, float]

        t = Table(data=None, names=names, dtype=dtype)
        for i in range(self.nsrc):
            if i % 100 == 0:
                print '%i:  z=%.3f' % (i, self.zz[i])

            if self.zz[i] < 1.5:
                src = Source(self.zz[i], self.LF1, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 1.5) & (self.zz[i] < 2.7):
                src = Source(self.zz[i], self.LF2, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 2.7) & (self.zz[i] < 3.4):
                src = Source(self.zz[i], self.LF3, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 3.4) & (self.zz[i] < 4.5):
                src = Source(self.zz[i], self.LF4, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 4.5) & (self.zz[i] < 5.5):
                src = Source(self.zz[i], self.LF5, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 5.5) & (self.zz[i] < 6.5):
                src = Source(self.zz[i], self.LF5, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 6.5) & (self.zz[i] < 7.5):
                src = Source(self.zz[i], self.LF6, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            elif (self.zz[i] >= 7.5) & (self.zz[i] <= 8.5):
                src = Source(self.zz[i], self.LF7, self.wave_array,
                             self.temp, self.add_lines, self.cosmo)

            # extinguished, attenuated template flux scaled to src luminosity
            wave, flux = src.process()

            mag = np.zeros(len(self.filters.filterlist), dtype=float)
            c = 2.99792e18
            for i,filt in enumerate(self.filters.filterlist):
                pivot, fluxden = self.filters.calc_synphot(wave,flux,src.z,filt)
                mag[i] = -2.5*np.log10(fluxden) - 2.5*np.log10(pivot**2) + \
                        2.5*np.log10(c) - 48.6 + 2.5*np.log10((1.+src.z)**2)
            
            # print output or add to table
            t.add_row([src.z, src.lf.zbase, src.lf.mst, src.lf.lst, src.lum, 
                       src.template, src.universe_age, src.age, 
                       self.add_lines, 0.0, # src.nly,
                       src.extstr, src.nclumps, src.ebv, src.tauv_clump,
                       mag[0], mag[1], mag[2], mag[3], mag[4]])

        t.sort('z')    
        t.write(self.outfits, format='fits')

        t['z'].format = '{:.3f}'
        t['Lst'].format = '{:.3e}'
        t['lum'].format = '{:.3e}'
        t['Uage'].format = '{:.3e}'
        t['age'].format = '{:.3e}'
        t['m606'].format = '{:.3f}'
        t['m814'].format = '{:.3f}'
        t['m110'].format = '{:.3f}'
        t['m160'].format = '{:.3f}'
        t['irac_ch1'].format = '{:.3f}'

        ascii.write(t, output=self.outdat, format='fixed_width_two_line',
                    position_char='=')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nsrc', type=int, default=8000,
                        help='number of sources to generate [default=8000]')
    parser.add_argument('--add_lines', action='store_true',
                        help='add emission lines to spectra?')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    cat = Catalog(args.nsrc, args.add_lines)


if __name__ == '__main__':
    main()
