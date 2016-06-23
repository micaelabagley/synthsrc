"""
.. module::
   :synopsis: Optional LFs.
"""
import os
import ConfigParser
import numpy as np
from scipy import integrate, interpolate

from utils import Cosmology, get_filesdir


class LuminosityFunction(object):
    """Class to calculate and initialize the luminosity function.

    The luminosity function is created from the Schechter paramters 
    in the config file, ``luminosityfunctions.cfg``, which can be found in 
    ``synthsrc/requiredfiles. 

    The cumulative LF is calculated by integrating the LF from :math:`10L^*`
    down to the luminosity corresponding to the provided magnitude limit,
    typically the :math:`1\sigma` magnitude limit of our survey. 

    Args:
        zbase (int): Average or central redshift of redshift range.
            Determines which LF to use.
        maglim (float): Magnitude limit for use as faint integration limit.
        log (Optional[str]): Filename of optional logfile, for keeping
            track of Schechter function parameters used in constructing 
            the cumulative LF.
    """    

    def __init__(self, zbase, maglim, log=None):
        """Initializes the LuminosityFunction class."""
        self.zbase = int(zbase)
        self.logfile = log

        self.reference = ''
        self.mst = 0.
        self.mwave = 0.
        self.phi = 0.
        self.alpha = 0.

        self.lst = 0.
        self.clf = 0
        self.invclf = None
        self.faintlim = 0.
        self.maglim = float(maglim[0])
        self.maglimwave = float(maglim[1])

        self.set_lf()


    def convert_mag(self, mag, wave, absmag=True):
        """Converts an AB magnitude to luminosity.

        Converts the input AB magnitude to :math:`L_{\lambda}`.
        If the input magnitude is an absolute magnitude, converts first
        to apparent magnitude:
        
        .. math::
           m = M + 5 \\ log(d[\mathrm{Mpc}] + 25 - 2.5 \\ log(1+z)

        Args:
            mag (float):
            wave (float):
            absmag (bool): 

        Returns:
            llambda (float): 
        """
        cosmo = Cosmology()
        dlum = cosmo.get_lumdist(self.zbase)
        pctocm = 3.0857e18
        mpctocm = 3.0857e24
        c = 2.99792e18

        if absmag:
            # convert absolute to apparent mag
            # M = m - 5log(d[Mpc]) - 25 + 2.5log(1+z)
            d_mpc = dlum / mpctocm
            mag = mag + 5.*np.log10(d_mpc) + 25. - 2.5*np.log10(1.+self.zbase)
        # AB mag zero point
        abzp = -2.5 * np.log10(3631.e-23)
        fnu = 10.**(-0.4*(mag + abzp))
       
        flambda = fnu * (c / wave**2)
        llambda = flambda * (4.*np.pi*dlum**2) * (1.+self.zbase)
        return llambda


    def cumulative_lf(self):
        """Calculates the cumulative LF for the given Schechter LF.

        Returns:
            clf (): cumulative LF normalized for use as a cumulative
                distribution function
            invclf (): inverse cumulative LF for use in randomly assigning
                synthetic sources a luminosity
        """
        # array of luminosities in units of lst
        x = np.arange(self.faintlim, 10, 0.001)

        schechter = lambda x,lst,phi,alpha: phi/lst * x**alpha * np.exp(-x)

        clf = np.zeros(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            clf[i] = integrate.quad(schechter, x[i], x[-1], 
                                    args=(self.lst, self.phi, self.alpha))[0]

        clf = clf / np.max(clf)
        # interpolate the inverse cumulative LF for random assignment of lums
        invclf = interpolate.interp1d(clf[::-1], x[::-1])

        if self.logfile is not None:
            log = open(self.logfile, 'a')
            log.write('\n<z> = %i\n' % self.zbase)
            log.write('   reference = %s\n' % self.reference)
            log.write('   mst = %.2f\n' % self.mst)
            log.write('   wave = %i\n' % self.mwave)
            log.write('   phi = %.2e\n' % self.phi)
            log.write('   alpha = %.2f\n' % self.alpha)
            log.write('   maglim = %.3f\n' % self.maglim)
            log.write('   maglim wave = %.3f\n' % self.maglimwave)
            log.write('   faintlim = %.3f\n' % self.faintlim)
            log.close()
        return clf, invclf


    def set_lf(self):
        """Sets the cumulative LF for the given redshift range.

        Reads in the Schechter parameters from the config file, converts 
        the :math:`M^*` into :math:`L^*`, and sets the faint luminosity 
        integration limit. Calls :func:`cumulative_lf` to calculate
        the cumulative LF. 
        """
        if not isinstance(self.zbase, basestring):
            zbase = str(self.zbase)
        else:
            zbase = self.zbase
        
        config = ConfigParser.ConfigParser()
        config.read(os.path.join(get_filesdir(),'luminosityfunctions.cfg'))
        options = config.options(zbase)
        for option in options:
            try:
                setattr(self, option, float(config.get(zbase, option)))
            except:
                setattr(self, option, config.get(zbase, option))
        
        # convert M* to L*
        mst = float(config.get(zbase, 'mst'))
        mwave = float(config.get(zbase, 'mwave'))
        self.lst = self.convert_mag(mst, mwave, absmag=True)

        # convert survey maglim to luminosity for integration limits
        wave = self.maglimwave / (1. + self.zbase)
        lum = self.convert_mag(self.maglim, wave, absmag=False)
        self.faintlim = lum / self.lst
        
        self.clf, self.invclf = self.cumulative_lf()


