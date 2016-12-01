import cosmolopy.distance as cd
import cosmolopy.constants as cc

class Cosmology(object):
    """A class to set the cosmology and calculate useful quantities.

    Args:
        h (Optional[float]): Hubble constant in units of 100 km/s/Mpc. 
            Defaults to 0.7
        omega_m (Optional[float]): Matter density parameter. Defaults 
            to 0.3
        omega_l (Optional[float]): Dark energy density parameter. 
            Defaults to 0.7
        setplanck (Optional[bool]): Sets the Planck 2015 cosmological 
            parameters instead of the defaults 
    """

    def __init__(self, h=0.7, omega_m=0.3, omega_l=0.7, setplanck=False):
        """Initializes the cosmology for use with cosmolopy.distance."""
        self.h = h
        self.omega_m = omega_m
        self.omega_l = omega_l

        if setplanck:
            self.set_planck2015()

        cosmo = {'omega_M_0':self.omega_m, 'omega_lambda_0':self.omega_l, \
                 'h':self.h}

        self.cosmo = cd.set_omega_k_0(cosmo)


    def set_planck2015(self):
        """Sets the Planck 2015 cosmological parameters.

        Hubble constant, matter and dark enery parameters are
        from `Planck 2015 results <https://arxiv.org/abs/1502.01589>`_

        - h = 0.678
        - omega_m = 0.308
        - omega_l = 0.692        
        """
        self.h = 0.678
        omega_m = 0.308
        omega_l = 0.692 
       

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

