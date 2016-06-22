import os
from glob import glob
import numpy as np
from astropy.io import fits
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

from utils import get_filtdir, get_templatedir, get_filesdir


class Filter(object):
    """A class to define a broadband filter curve.

    The filter transmission curve is read in, normalized, and interpolated
    onto a finer-resolution wavelength array. 
    In :meth:`synthsrc.Filter.calc_synphot`, the spectrum is interpolated 
    to the same wavelength array. The spectrum and all filters used in
    calcuating the broadband photometry therefore have the same wavelength
    resolution.

    Filter transmission curves are stored in the ``synthsrc/filters`` 
    directory and are called ``[filter].dat``. Available filters are:

    - `HST`/WFC3 

        - F606W 
        - F814W 
        - F110W 
        - F160W 

    - `Spitzer`/IRAC 

        - CH1, :math:`3.6 \mu m`

    Args:
        wave_array (float): Wavelength array for interpolation of 
            filters and SED templates
    """

    def __init__(self, wave_array):
        """Initializes the Filter class."""
        self.filterdir = get_filtdir()
        self.filterlist = ['F606W', 'F814W', 'F110W', 'F160W', 'IRAC_CH1']

        self.wave_array = wave_array

        for filt in self.filterlist:
            setattr(self, filt, self.read_filt(filt))


    def interp_spectrum(self, wave, flux):
        """Interpolates spectrum to a finer resolution wavelength array.

        Values outside the range of the input wavelength array are 
        set to zero.

        Args:
            wave (float): Input wavelength array
            flux (float): Input flux array

        Returns:
            float: Flux array interpolated onto higher-res wavelength array
        """
        f = interpolate.interp1d(wave, flux, bounds_error=False, fill_value=0.)
        return f(self.wave_array)


    def read_filt(self, filt):
        """Reads in the filter transmission curve.
    
        :func:`read_filt` normalizes the transmission curve and calls 
        :func:`interp_spectrum` to interpolate it to a finer resolution 
        wavelength array.

        Args:
            filt (str): Filter ID

        Returns:
            transmission (float): Interpolated filter transmission curve
        """
        d = np.genfromtxt(os.path.join(self.filterdir, filt+'.dat'))
        wave = d[:,0]
        flux = d[:,1]

        norm = integrate.simps(flux)
        flux = flux / norm

        # interpolate transmission curve to a finer wavelength resolution 
        transmission = self.interp_spectrum(wave, flux) 

        return transmission


    def calc_synphot(self, wave, flux, redshift, filt, plotphot=False):
        """Calculates the flux density of a spectrum in a broad band filter.

        The mean flux density in a broad passband, :math:`P(\lambda)`, 
        is defined as:

        .. math::
        
           f_{\lambda}(P) = \\frac{\int P_{\lambda} \\ f_{\lambda} \\ \lambda \\ \mathrm{ d}\lambda}{\int P_{\lambda} \\ \lambda \\ \mathrm{ d}\lambda}
        
        The pivot wavelength of the filter is then:
        
        .. math::
           \lambda_p(P) = \sqrt{\\frac{\int P(\lambda) \lambda \mathrm{ d}\lambda}{\int P(\lambda) \mathrm{ d}\lambda / \lambda}}

        (See the `Synphot manual <http://www.stsci.edu/institute/software_hardware/stsdas/synphot/SynphotManual.pdf>`_)

        Args:
            wave (float): Rest frame wavelength array of spectrum
            flux (float): Flux array of spectrum
            redshift (float): Source redshift
            filt (str): Filter ID
            plotphot (Optional[bool]): Option to plot and display the 
                spectrum, filter transmission curve, and resulting
                photometry. Defaults to False.

        Returns:
            pivot (float): The pivot wavelength of the filter
            flux_density (float): The mean flux density calculated for the broadband filter
        """
        # redshift spectrum
        lobs = wave * (1. + redshift)

        # interpolate spectrum to filter's wavelength resolution
        spectrum = self.interp_spectrum(lobs, flux)

        transmission = getattr(self, filt)

        func1 = transmission * spectrum * self.wave_array
        func2 = transmission * self.wave_array
        func3 = transmission / self.wave_array

        flux_density = integrate.simps(func1) / integrate.simps(func2)

        # pivot wavelength of filter
        pivot = np.sqrt(integrate.simps(func2) / integrate.simps(func3))

        if plotphot:
            plt.plot(self.wave_array, spectrum, 'k')
            filtscale = np.max(spectrum) / np.max(transmission)
            plt.plot(self.wave_array, transmission*filtscale, 'g')
            plt.scatter(pivot, flux_density, marker='o', s=150, color='r')
            plt.show()

        return pivot, flux_density


    def plot_filters(self):
        """
        """
        pass


class Template(object):
    """A class to read in BC03 templates.

    BC03 templates are stored in the ``synthsrc/highztemplates`` directory 
    and are called ``[IMF]_[metal]_[SFH].fits``, where 

    - IMF is either `salp` or `chab` for Salpeter and Chabrier initial
      mass functions, respectively; 

    - metal is the metallicity ranging from `m22` to `m62` (solar);

    - SFH is one of `tau0p01`, `tau0p50`, and `tau5p00`, for exponentially
      declining star formation histories with characteristic timescales
      of 0.01, 0.5, and 5 Gyr, respectively.
    """

    def __init__(self):
        """Initializes the Template class."""
        self.tempdir = get_templatedir()
                

        # get array of available ages
        data = fits.getdata(os.path.join(self.tempdir,'salp_m22_tau0p01.fits'))
        cols = data.columns.names
        self.ages = np.array([float(x) for x in cols if x != 'wave'])

        self.set_templates()

        # available emission lines
        self.lines = fits.getdata(os.path.join(get_filesdir(),
                                               'line_ratio_table.fits'))


    def set_templates(self):
        """Reads in each template for easy access later.

        Each template FITS table is stored along with the number of 
        ionizing photons created as a function of age (in the ``*.3color``
        BC03 output files)
        """
        templates = glob(os.path.join(self.tempdir, '*m*tau*.fits'))
        for template in templates:
            temp = os.path.splitext(os.path.basename(template))[0]
            nly = os.path.splitext(os.path.basename(template))[0] + '_nly'
            setattr(self, temp, fits.getdata(template))

            # read in *.3color for 
            data = np.genfromtxt(template.replace('.fits','.3color'),
                                 usecols=(0,5))
            setattr(self, nly, data)

