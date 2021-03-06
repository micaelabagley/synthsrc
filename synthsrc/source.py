import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

from attenuation import *


class Source(object):
    """Class to generate the properties and spectrum of a synthetic source.

    The source is randomly assigned a BC03 spectral template
    (:func:`choose_template`) and a luminosity, age, and dust extinction
    (:func:`generate_properties`). Emission lines may be added to the
    spectrum (:func:`add_emission_lines`). 
    The ISM extinction and IGM attenuation are applied and the spectrum 
    is scaled to the assigned luminosity (:func:`process`).

    BC03 Templates are from :class:`~.templates.Template`, ISM extincation 
    is from :class:`~.attenuation.ISMExtinction` and IGM attenuation is 
    from :class:`~.attenuation.IGMAttenuation`.

    Args:
        z (float): Redshift of source
        LF (synthsrc.lfs.LuminosityFunction): Instance of 
            :class:`~.lfs.LuminosityFunction` corresponding to source redshift
        wave_array (float): High-resolution wavelength array used for 
            interpolation of the filters and SED
        temp (synthsrc.templates.Template): Instance of 
            :class:`~.templates.Template` 
        add_lines (bool): If ``True``, emission lines are added to the 
            spectrum using :func:`add_emission_lines`
        cosmo (synthsrc.utils.Cosmology): Instance of 
            :class:`~.utils.Cosmology`

    """

    def __init__(self, z, LF, wave_array, temp, add_lines, cosmo):
        self.wave_array = wave_array

        # source properties
        self.imf = ''
        self.metallicity = ''
        self.sfh = ''
        self.lumlst = 0.
        self.lum = 0.
        self.age = 0.
        self.ext_law = 0
        self.extstr = ''
        self.ebv = 0.
        self.nclumps = 0
        self.tauv_clump = 0.

        # reading in BC03 templates
        self.temp = temp
        self.ages = self.temp.ages
        self.add_lines = add_lines

        # source redshift and luminosity distance
        self.z = z
        self.dlum = cosmo.get_lumdist(z)

        # age of universe at source redshift
        self.universe_age = cosmo.get_lookback_time(z)

        # luminosity function for source redshift
        self.lf = LF

        self.template = self.choose_template()

        # possible E(B-V)
        ebvs = np.arange(0, 0.51, 0.01)
        self.generate_properties(ebvs)


    def find_thing(self, array, value):
        """Returns the array index of the element closest to the given value."""
        return np.argmin(np.abs(array - value))


    def choose_template(self):
        """Randomly chooses a BC03 template.
    
        Randomly chooses an IMF, metallicity, and SFH for the synthetic
        SED and uses these to construct the template filename that will 
        provide the source's spectrum. Templates are in 
        ``synthsrc/highztemplates`` with the naming convention
        ``[IMF]_[metal]_SFR].fits``, where

        - IMF is the initial mass function, currently either Salpeter 
          or Chabrier;

        - metal is the metallicity, ranging from `m22` (:math:`Z=0.0001`)
          to `m62` (:math:`Z=0.02`, solar);

        - SFH is the star formation history, where all options are 
          exponentially declining with characteristic timescales of 
          :math:`\\tau=0.01`, :math:`0.5`, and :math:`5` Gyr.

        Returns:
            template (str): Filename of BC03 template 

        """
        # --- imf ---
        imfs = ['chab', 'salp']
        self.imf = imfs[np.random.randint(2)]

        # --- metallicity ---
        if self.z <= 1:
            metals = ['m22', 'm32', 'm42', 'm52', 'm62']
        elif (self.z >1 ) & (self.z <= 2):
            metals = ['m22', 'm32', 'm42', 'm52']
        elif (self.z > 2) & (self.z <= 3):
            metals = ['m22', 'm32', 'm42']
        else:
            metals = ['m22', 'm32']
        self.metallicity = metals[np.random.randint(len(metals))]

        # --- sfh ---
        sfhs = ['tau0p01', 'tau0p50', 'tau5p00']
        self.sfh = sfhs[np.random.random_integers(0,2)]

        return '_'.join([self.imf, self.metallicity, self.sfh])
   

    def generate_properties(self, ebvs):
        """Assigns source a luminosity, age, and dust extinction.

        The source luminosity is randomly pulled from the LF appropriate
        for the source's redshift. The inverse cumulative LF (created by
        :func:`~lfs.LuminosityFunction.cumulative_lf`) is indexed with a 
        number from 0 to 1 randomly generated by :func:`numpy.random.random`.
        
        The age of the source can range from 10 Myr to the age of the 
        universe at the source's redshift and is pulled randomly from 
        those available in the BC03 template.

        There are two possible geometries for dust in the ISM of the 
        synthetic galaxy:

        1. a uniform slab of dust in front of the source, or
        2. a clumpy slab in front of an extended source.

        In the first case, extinction follows the classical
        :math:`I_o/I_i = e^{-\\tau_{\lambda}}`, where :math:`I_o` and
        :math:`I_i` are the observed and intrinsic intensities, 
        respectively. The  `Calzetti et al. (2000) 
        <http://adsabs.harvard.edu/abs/2000ApJ...533..682C>`_ law for 
        starburst galaxies is used in this case 
        (:func:`~.attenuation.ISMExtinction.calz`). 

        For the second geometry, all clumps are assumed to have the same
        optical depth, :math:`\\tau_{c,\lambda}`, which is pulled 
        randomly from :math:`0.1 \leq \\tau_{c,\lambda} \leq 10`.
        Each clump acts as a uniform dust screen and follows the 
        `Cardelli, Clayton, and Mathis (1989) 
        <http://adsabs.harvard.edu/abs/1989ApJ...345..245C>`_ extinction
        law (:func:`~.attenuation.ISMExtinction.ccm`). The ISM 
        extinction then goes as 
        :math:`I_o/I_i = \mathrm{exp}[-N(1-e^{-\\tau_{c,\lambda}})]`,
        where :math:`N` is the average number of clumps along the 
        line of sight. (See `Natta & Panagia (1984) 
        <http://adsabs.harvard.edu/abs/1984ApJ...287..228N>`_). 
        :math:`N` can be 1, 2, 5, or 10 clumps. 

        In both cases, the maximum available extinction is restricted
        following `Hayes et al. (2011) 
        <http://adsabs.harvard.edu/abs/2011ApJ...730....8H>`_.

        Args:
            ebvs (float): an array of possible :math:`E(B-V)`

        """
        # --- luminosity ---
        # assign a luminosity from the cumulative LF (in units of L*)
        self.lumlst = self.lf.invclf(np.random.random())
        self.lum = self.lumlst * self.lf.lst

        # --- age ---
        # find template age closest to age of universe 
        wage = self.find_thing(self.ages, self.universe_age)
        # template age cannot be > age of universe
        if self.ages[wage] > self.universe_age:
            wage -= 1
        # randomly assign source age from a uniform distribution
        self.age = self.ages[np.random.random_integers(0, wage)]

        # --- dust ---
        # Calzetti or Cardelli extinction laws
        self.ext_law = np.random.random_integers(0,1)
        # dust evolution with redshift: ebv = 0.386 * exp(-z / 3.42)
        # (Hayes et al. 2011, ApJ, 730, 8)
        med_ebv = 0.386 * np.exp(-self.z / 3.42)

        if self.ext_law == 0:
            # Calzetti extinction law only requires ebv
            # randomly assign an extinction not to exceed that allowed by
            # the evolution from Matt's paper
            webv = self.find_thing(ebvs, med_ebv)
            if ebvs[webv] > med_ebv:
                webv -= 1
            self.ebv = ebvs[np.random.random_integers(0, webv)]
    
        elif self.ext_law == 1:
            # Cardelli law requires nclumps and optical depth per clump
            # assume number of clumps
            clumps = np.array([1, 2, 5, 10])
            self.nclumps = clumps[np.random.random_integers(0,3)]
            # assume optical depth for clumps in the V band
            tauv_clumps = ebvs
            # calculate effective optical depth in V band
            tauv_eff = self.nclumps * (1. - np.exp(-tauv_clumps))
            # E_{B-V} = AV / RV = (1.086 * tauv_eff) / RV
            ebv_eff = (1.086 * tauv_eff) / 3.1
            # randomly assign clump optical depth such that the effective
            # optical depth does not exceed the allowed value
            webv = self.find_thing(ebv_eff, med_ebv)
            if ebv_eff[webv] > med_ebv:
                webv -= 1
            self.tauv_clump = tauv_clumps[np.random.random_integers(0, webv)]
            
           
    def process(self, plotcheck=False):
        """Processes the template SED of source.

        Applies ISM extinction and IGM attenuation to the input template
        spectrum. Scales the full spectrum so that the flux at 
        :math:`\lambda = \lambda_{M^*}` is the flux randomly 
        assigned from the cumulative LF. This is done using the average
        flux within a range :math:`\pm 50 \unicode{x212B}` from 
        :math:`\lambda_{M^*}`.

        Args:
            plotcheck (Optional[bool]): Set to ``True`` to plot the spectrum
                at each stage as a check. Default is ``False``.

        Returns:
            (tuple): Tuple containing:
        
                wave (float): Wavelength array

                outflux (float): Flux array
        """
        #wave, templateflux, nly = self.read_template()
        wave, templateflux = self.read_template()

        # apply ISM extinction
        ism = ISMExtinction()
        if self.ext_law == 0:
            # Calzetti et al. 2000
            reddenedflux = ism.calz(wave, templateflux, self.ebv)
            self.extstr = 'calz'
        elif self.ext_law == 1:
            # Cardelli, Clayton & Mathis (CCM) 1989  - clumpy ISM
            # the optical depth of the clumps as a function of wavelength
            # will follow the CCM extinction law

#            # following Natta & Panagia 1984, ApJ, 287, 228
#            # I_obs/I_init = exp(-nclumps*(1-exp(-tau_clumps))) = exp(-tau_eff)
#            # calculate effective optical depth in V band
#            tauv_eff = self.nclumps * (1. - np.exp(-self.tauv_clumps))
#            # E_{B-V} = AV / RV = (1.086 * tauv_eff) / RV,  RV = 3.1
#            ebv_eff = (1.086 * tauv_eff) / 3.1
            
            taulam_clump = ism.ccm(wave, self.tauv_clump)
            # redden according to Natta & Panagia 1984, Scarlata et al. 2009
            factor = -self.nclumps * (1. - np.exp(-taulam_clump))
            reddenedflux = templateflux * np.exp(factor)
            self.extstr = 'ccm'

        # IGM attenuation from Inoue et al.
        igm = IGMAttenuation()
        ext = igm.inoue(wave, self.z)
        extflux = reddenedflux * ext
        
        # normalize spectrum to wavelength at which LF is calculated
        flux = self.lum / (4. * np.pi * self.dlum**2 * (1.+self.z))
        # scale full template SED so that the flux at lambda=self.lf.mwave
        # is the flux randomly assigned from the cumulative LF

        # find flux at LF wavelength
        # taken an average in a bin +/- 50 A from center
        wlam = self.find_thing(wave, self.lf.mwave)
        wlam1 = self.find_thing(wave, self.lf.mwave-250.)
        wlam2 = self.find_thing(wave, self.lf.mwave+250.)
        avgflux = np.mean(extflux[wlam1:wlam2])
        
        # rescale
        scale = flux / avgflux
        outflux = extflux * scale

        if plotcheck:
            # input spectrum
            plt.plot(wave, templateflux)
            plt.xlim(0,30000)
            # reddened and attenuated spectrum
            plt.plot(wave, reddenedflux, '#ffa500')
            plt.plot(wave, extflux, 'r')
            # wavelength range used for flux normalization
            plt.vlines([wave[wlam], wave[wlam1], wave[wlam2]], 0, 1.e-20)
            plt.hlines(avgflux, wave[wlam1], wave[wlam2])
            # output spectrum
            plt.plot(wave, outflux, 'r--')
        
        return wave, outflux
        

    def read_template(self):
        """Reads in the BC03 template to get the spectrum for the given age.

        The source template is chosen via :func:`choose_template`, and 
        the age is assigned by :func:`generate_properties`. Emission
        lines are added via :func:`add_emission_lines` if
        :attr:`Source.add_lines` is ``True``.

        Returns:
            (tuple): tuple containing:
            
                wave_array (float): Wavelength array

                flux (float): Flux array
        """
        data = getattr(self.temp, self.template)
        tempflux = data['%.3e'%self.age]
        wave = data['wave']

        # interpolate onto higher resolution wavelength array
        f = interpolate.interp1d(wave, tempflux, bounds_error=False,
                                 fill_value=0.)
        templateflux = f(self.wave_array)

        # put spectrum in correct units - BC03 flux output in arbitrary units
        # - multiply by Lsol to get ergs/s
        # - scale by mass? (divide by the galaxy mass at the appropriate
        #                   model age so all spectra are normalized to 1 Msol)
        # - put at redshift of source
        lsol = 3.839e33

        if self.add_lines:
            templateflux = self.add_emission_lines(self.wave_array,
                                                   templateflux)

        lum = templateflux * lsol
        flux = lum / (4. * np.pi * self.dlum**2 * (1.+self.z))

        return self.wave_array, flux


    def add_emission_lines(self, wave, flux):
        """Adds emission lines to the synthetic spectrum.

        Emission lines are modeled as Gaussian profiles with FWHM of 
        :math:`\sim 20\unicode{x212B}`. The :math:`\mathrm{H}\\beta` 
        line luminosity is determined by the flux of hydrogen-ionizing 
        photons output by the BC03 models. Emission line coefficients for 
        :math:`\mathrm{H}\\alpha` and :math:`\mathrm{Ly}\\alpha` are taken 
        from Table 1 of `Schaerer (2003) 
        <http://adsabs.harvard.edu/abs/2003A%26A...397..527S>`_.
        All other line ratios, which depend on the model's metallicity, 
        are taken from `Anders & Fritze-v. Alvensleben (2003) 
        <http://adsabs.harvard.edu/abs/2003A%26A...401.1063A>`_ assuming
        an electron density of :math:`n_e=100 \\ \mathrm{cm}^{-3}`
        and temperature :math:`T_e = 10000` K.

        Args:
            wave (float): Wavelength array in :math:`\unicode{x212B}`
            flux (float): Flux array

        Returns:
            flux (float): Flux array with emission lines added
        """
        lsol = 3.839e33
        data = getattr(self.temp, self.template+'_nly')
        logages = data[:,0]
        logq = data[:,1]

        wage = self.find_thing(10.**logages, self.age)
        self.nly = 10.**logq[wage]

        # get wavelength and line luminosities for
        # Hbeta
        hbeta_wave = 4861.
        # worked out using recombination coefficients from Osterbrock
        # also Anders 2003 (A&A, 401, 1063)
        hbeta_lum = 4.757e-13 * self.nly / lsol

        # Schaerer 2003 (A&A, 397, 527), Table 1
        #   Case B assumed:
        #       T_e=30000K for Z<\=10e-5, else T_e=10000K
        #       n_e=10^3 cm^-3
        #   Assume 0.68 photons are converted to Lya
        #   L_1 = c_1 * (1 - f_esc) * Q
        #       Ha: c_1 = 1.37e-12
        #       Lya: c_1 = 1.04e-11
        #   So assuming f_esc = 0
        # Halpha
        halpha_wave = 6563.
        halpha_lum = 1.37e-12 * self.nly / lsol
        # Lya
        lya_wave = 1216.
        lya_lum = 1.04e-11 * self.nly / lsol

        # line ratios will depend on the metallicity
        if (self.metallicity == 'm22') | (self.metallicity == 'm32'):
            ratios = self.temp.lines['m32_ratio'][0]
        elif self.metallicity == 'm42':
            ratios = self.temp.lines['m42_ratio'][0]
        else:
            ratios = self.temp.lines['m52_62_72_ratio'][0]

        waves = np.append([lya_wave, halpha_wave, hbeta_wave], 
                          self.temp.lines['lambda'][0])
        lines = np.append([lya_lum, halpha_lum, hbeta_lum],
                          ratios*hbeta_lum)
    
        a = 1.
        sig = 3.
        for i in range(lines.shape[0]):
            line_center = self.find_thing(wave, waves[i])
            
            gauss = a * np.exp(-(wave-wave[line_center])**2 / (2.*sig**2))

            # normalize line profile so line luminosity is total lum in line
            norm = integrate.simps(gauss)
            if norm != 0:
                profile = gauss * (lines[i] / norm)
                flux += profile

        return flux
