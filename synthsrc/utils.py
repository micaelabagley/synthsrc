import os
from glob import glob
from datetime import datetime
import numpy as np
import cosmolopy.distance as cd
import cosmolopy.constants as cc

import synthsrc
#from catalog import Catalog
#from randomize import RandomCatalog

class Simulation(object):
    """A class to control input and output of the synthsrc simulations.

    Args:
        nsrc (int): 
        add_lines (bool): 
        sig (float): 
        nrealz (Optional[int]):
        outdir (Optional[str]):

    """
    def __init__(self, nsrc, add_lines, sig, nrealz=10, phottype='APER', limtype='med', wispcat=None, outdir='.', errorplot=True):
        """Initialize"""
        self.nsrc = nsrc
        self.add_lines = add_lines

        self.outputs = os.path.join(outdir, 'outputs')
        self.plots = os.path.join(outdir, 'plots')
        # create outputs and plots if necessary
        if not os.path.exists(self.outputs):
            os.mkdir(self.outputs)
        if not os.path.exists(self.plots):
            os.mkdir(self.plots)

        # synthetic catalog
        self.outcat = os.path.join(self.outputs, 'synthcat_%i'%nsrc)
        if len(glob('%s.fits' % self.outcat)) != 0:
            today = datetime.today()
            date = '_%i.%i.%i' % (today.month,today.day,today.year)
            time = '%i.%i.%i' % (today.hour, today.minute, today.second)
            out = date + '-' + time
            self.outcat = os.path.splitext(self.outcat)[0] + out

        self.outfits = self.outcat + '.fits'
        self.outdat = self.outcat + '.dat'

        self.logfile = self.outcat + '.log'
        log = open(self.logfile, 'w')
        log.write('synthcat\n========\n\n%i sources\n' % nsrc)
        log.write('output: %s\n' % self.outcat)
        log.close()

        cat = synthsrc.Catalog(nsrc, add_lines, self.outfits, self.outdat, 
                      self.logfile)

        # randomizing the catalog
        #self.sig = sig
        #self.nrealz = nrealz
        #self.phottype = phottype
        #self.limtype = limtype
        randcat = self.outcat + '_randomized_%.1f.fits'%sig
    
        rcat = synthsrc.RandomCatalog(self.outfits, sig, nrealz, randcat, phottype,
                             limtype, self.plots, wispcat)


class Cosmology(object):
    """A class to set the cosmology and calculate useful quantities.

    Input parameters (Hubble constant, matter and dark energy 
    density parameters) are based on the `Planck 2015 results 
    <https://arxiv.org/abs/1502.01589>`_

    Args:
        h (Optional[float]): Hubble constant in units of 100 km/s/Mpc. 
            Defaults to 0.678
        omega_m (Optional[float]): Matter density parameter. Defaults 
            to 0.308
        omega_l (Optional[float]): Dark energy density parameter. 
            Defaults to 0.692
        log (Optional[str]): Filename of optional logfile, for keeping
            track of cosmology parameters used in calculating luminosity
            distance, etc.
    """

    def __init__(self, h=0.678, omega_m=0.308, omega_l=0.692, log=None):
        """Initializes the cosmology for use with cosmolopy.distance."""
        self.h = h
        self.omega_m = omega_m
        self.omega_l = omega_l
        cosmo = {'omega_M_0':self.omega_m, 'omega_lambda_0':self.omega_l, \
                 'h':self.h}

        if log is not None:
            f = open(log, 'a')
            f.write('\nCosmological Parameters\n')
            f.write('   Omega_M = %.2f\n' % self.omega_m)
            f.write('   Omega_L = %.2f\n' % self.omega_l)
            f.write('   h = %.2f\n' % self.h)
            f.close()

        self.cosmo = cd.set_omega_k_0(cosmo)


    def get_lumdist(self, z):
        """Calculates the luminosity distance to redshift z.

        Uses :func:`cosmolopy.distance.luminosity_distance`, 
        which is based on David Hogg's `Distance measures in cosmology.
        <https://arxiv.org/abs/astro-ph/9905116>`_

        Args:
            z (float): Redshift 

        Returns:
            float: Luminosity distance in cm
        """
        # luminosity distance in Mpc
        dlum_mpc = cd.luminosity_distance(z, **self.cosmo) 
        return dlum_mpc * cc.Mpc_cm

    
    def get_lookback_time(self, z):
        """Calculates the lookback time to redshift z.

        Uses :func:`cosmolopy.distance.age` and 
        :func:`cosmolopy.distance.lookback_time`, 
        which are based on equation 30 of David Hogg's `Distance 
        measures in cosmology. <https://arxiv.org/abs/astro-ph/9905116>`_
           
        Args:
            z (float): Redshift for which to calculate the lookback time

        Returns:
            (float): Lookback time in years

        """
        sec = cd.age(z, **self.cosmo)
        return sec / cc.yr_s 


def get_filesdir():
    """Returns the directory containing other required files."""
    return os.path.abspath(os.path.join(synthsrc.__path__[0], os.pardir, 
                                        'requiredfiles'))


#def get_outputdir():
#    """Returns the directory contained"""
#    pass


def get_filtdir():
    """Returns the directory containing filter transmission curves."""
    return os.path.abspath(os.path.join(synthsrc.__path__[0], os.pardir,
                                        'filters'))


def get_templatedir():
    """Returns the directory containing the BC03 spectral templates."""
    return os.path.abspath(os.path.join(synthsrc.__path__[0], os.pardir,
                                        'highztemplates'))


def hist_bins_scotts(data):
    """Use Scott's rule to determine histogram bin size.

    Scott's rule is best for random samples of normally distributed data.

    Args:
        data (float): Array of data to be histogrammed

    Returns:
        binsize (float): Width of bins
        nbins (int): Number of bins
        bins (float): Array of bin edges
    """
    Nsrc = data.shape[0]
    sig = np.std(data)
    binsize = 3.5*sig / Nsrc**(1./3.)
    bins = np.arange(np.min(data), np.max(data), binsize)
    nbins = bins.shape[0]
    return binsize, nbins, bins


def freedman_diaconis_bins(data):
    """Use Freedman-Diaconis rule to determine histogram bin size.

    The interquartile range (IQR) is less sensitive to outliers

    Args:
        data (float): Array of data to be histogrammed

    Returns:
        binsize (float): Width of bins
        nbins (int): Number of bins
        bins (float): Array of bin edges
    """
    Nsrc = data.shape[0]
    q75,q25 = np.percentile(data, [75,25])
    iqr = q75 - q25
    binsize = 2. * iqr / Nsrc**(1/3.)
    bins = np.arange(np.min(data), np.max(data), binsize)
    nbins = bins.shape[0]
    return binsize, nbins, bins


def plot_lfs():
    """
    """
    pass
