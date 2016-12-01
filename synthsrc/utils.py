import os
from glob import glob
from datetime import datetime
import numpy as np
import cosmolopy.distance as cd
import cosmolopy.constants as cc

import synthsrc
#from catalog import Catalog
#from randomize import RandomCatalog


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
