import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from utils import get_filesdir


class ISMExtinction(object):
    """Class to calculate and apply extinction from the interstellar medium.

    Options are:

    - :func:`calz` - Calzetti et al. (2000)
    - :func:`ccm` - `Cardelli, Clayton, and Mathis (1989)
    - :func:`allen` - Allen (1976)
    - :func:`prevot` - Prevot et al. (1984)
    - :func:`seaton` - Seaton (1979)
    - :func:`fitz` - Fitzpatrick (1986)

    .. note:: 

    For all reddening laws, supplying a negative ebv will deredden fluxes.
    """

    def __init__(self):
        self.calz_klam = 0
        self.ccm_klam = 0
        self.allen_klam = 0
        self.prevot_klam = 0
        self.seaton_klam = 0
        self.fitz_klam = 0
        self.wild_klam = 0

        self.calz_RV = 0
        self.ccm_RV = 0
        self.allen_RV = 0
        self.prevot_RV = 0
        self.seaton_RV = 0
        self.fitz_RV = 0
        self.wild_RV = 0


#    def apply_extinction(self, wave, flux, ebv, law):
#        """
#        """
#        # set extinction, get extinction law and RV
#        getattr(self, law)(wave, flux, ebv)
#        klam = getattr(self, law+'_klam')
#        RV = getattr(self, law+'_RV')
#
#        if law == 'ccm':
#            AV = 1.086 * 
#        return flux * 10.0**(-0.4*klam*ebv)


    def calz(self, wave, flux, ebv, RV=4.05):
        """Applies the reddening law for starburst galaxies from `Calzetti 
        et al. (2000) <http://adsabs.harvard.edu/abs/2000ApJ...533..682C>`_.

        The Calzetti law is valid for galaxy spectra where massive stars 
        dominate the radiation output and between 0.12 and 2.2 
        :math:`\mu m`.
        :func:`calz` is based on the IDL procedure :func:`CALZ_UNRED` 
        written by W. Landsman, which extrapolates between 0.12 and 0.0912
        :math:`\mu m`. Here, the wavelength range is extended below 0.0912. 
    
        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). Should be that derived for 
                the stellar continuum, :math:`E(B-V)_{stars}`, which is 
                related to the reddening derived from the gas, 
                :math:`E(B-V)_{gas}`, via the Balmer decrement: 
                :math:`E(B-V)_{stars} = 0.44 * E(B-V)_{gas}`.
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 4.05

        Returns:
            reddenedflux (float): Spectrum reddened according to the Calzetti law.
        """
        w1 = np.where((wave >= 6300) & (wave <= 22000))
        # extend Calzetti reln to lambda < 6300A
        w2 = np.where(wave < 6300)

        # inverse wavelength, um^-1
        x = 10000. / wave

        # reddening curve
        klam = np.zeros(flux.shape[0], dtype=float)
        # 0.63 um <= lambda <= 2.20 um
        if klam[w1].shape[0] != 0:
            klam[w1] = 2.659 * (-1.857 + 1.040*x[w1]) + RV

        # lambda < 0.63 um
        if klam[w2].shape[0] != 0:
            p = np.poly1d([0.011, -0.198, 1.509, -2.156])
            klam[w2] = 2.659*p(x[w2]) + RV

        self.calz_klam = klam
        self.calz_RV = RV
        
        return flux * 10.0**(-0.4*klam*ebv)


    def ccm(self, wave, tauV, RV=3.1):
        """Applies the reddening law from `Cardelli, Clayton, and Mathis (1989) <http://adsabs.harvard.edu/abs/1989ApJ...345..245C>`_.

        The reddening law includes the update for the near-UV given by 
        `O'Donnell (1994) <http://adsabs.harvard.edu/abs/1994ApJ...422..158O>`_.
        :func:`ccm` is based on the IDL procedure :func:`CCM_UNRED`
        written by W. Landsman. 
        Parameterization is valid from the far-UV to the IR 
        (0.1 to 3.5 :math:`\mu m`).

        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). 
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 3.1

        Returns:
            reddenedflux (float): Spectrum reddened according to the CCM law.

        """
        # inverse wavelength, um^-1
        x = 10000. / wave
        a = np.zeros(x.shape[0], dtype=float)
        b = np.zeros(x.shape[0], dtype=float)

        # IR
        ir = np.where((x > 0.3) & (x < 1.1))
        if x[ir].shape[0] != 0:
            a[ir] = 0.574 * x[ir]**1.61
            b[ir] = -0.527 * x[ir]**1.61

        # Optical/NIR
        # original coefficients from CCM89
        '''
        c1 = [1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, \
              -0.77530, 0.32999]         
        c2 = [0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, \
              5.30260, -2.09002]
        #'''
        # new coefficients from O'Donnell (1994)
        c1 = [ 1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
        c2 = [ 0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]
        p1 = np.poly1d(c1[::-1])
        p2 = np.poly1d(c2[::-1])

        op = np.where((x >= 1.1) & (x < 3.3))
        if x[op].shape[0] != 0:
            y = x[op] - 1.82
            a[op] = p1(y)
            b[op] = p2(y)

        # Mid-UV
        muv = np.where((x >= 3.3) & (x < 8))
        if x[muv].shape[0] != 0:
            y = x[muv]
            Fa = np.zeros(y.shape[0], dtype=float)
            Fb = np.zeros(y.shape[0], dtype=float)
            w1 = np.where(y > 5.9)
            if y[w1].shape[0] != 0:
                y1 = y[w1] - 5.9
                Fa[w1] = -0.04473 * y1**2 - 0.009779 * y1**3
                Fb[w1] = 0.2130 * y1**2 + 0.1207 * y1**3

            a[muv] = 1.752 - 0.316*y - (0.104 / ((y-4.67)**2 + 0.341)) + Fa
            b[muv] = -3.090 + 1.825*y + (1.206 / ((y-4.62)**2 + 0.263)) + Fb

        # Far-UV
        c1 = [-1.073, -0.628, 0.137, -0.070]
        c2 = [13.670, 4.257, -0.420, 0.374]
        p1 = np.poly1d(c1[::-1])
        p2 = np.poly1d(c2[::-1])
        #fuv = np.where((x >= 8) & (x <= 11))
        # extended down to lambda ~ 80A
        fuv = np.where((x >= 8) & (x <= 125))
        if x[fuv].shape[0] != 0:
            y = x[fuv] - 8.
            a[fuv] = p1(y)
            b[fuv] = p2(y)

        AV = 1.086 * tauV
        klam = a + b/RV
        Alambda = AV * klam

        self.ccm_klam = klam * RV
        self.ccm_RV = RV
        # return the optical depth, tau_lambda
        return Alambda / 1.086


    def allen(self, wave, flux, ebv, RV=3.1):
        """Applies the reddening law from `Allen (1976) 
        <http://adsabs.harvard.edu/abs/1976asqu.book.....A>`_.

        The reddening law:

        .. math::
           \\frac{A_{\lambda}}{A_V} = \\frac{k_{\lambda}}{R_V}

        is taken from a table of absorption from 0.1 to 10 :math:`\mu m` 
        in the bands U, B, V and I. The absorption :math:`A_{\lambda}` 
        is normalized to :math:`A_V = 1.0`.
        
        Interpolation is required for wavelengths between those in 
        the table. Extrapolation for longer and shorter wavelengths
        in not implemented at this time.

        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). 
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 3.1

        Returns:
            reddenedflux (float): Spectrum reddened according to the Allen law.

        """
        # reddening law, A_lam / A_V = k_lam / R_V
        lam = np.array([1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500.,
                        2850., 3330., 3650., 4000., 4400., 5000., 5530., 6700.,
                        9000., 10000., 20000., 100000.])
        klam = np.array([4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 
                         1.97, 1.67, 1.58, 1.45, 1.32, 1.13, 1.00, 0.74, 
                         0.46, 0.38, 0.11, 0.00])

        # interpolate in between wavelengths
        spline = InterpolatedUnivariateSpline(lam, klam)
        klam_interp = spline(wave)

        self.allen_klam = klam_interp * RV
        self.allen_RV = RV
        
        return flux * 10.0**(-0.4*klam_interp*ebv)


    def prevot(self, wave, flux, ebv, RV=2.72):
        """Applies the reddening law for the SMC from `Prevot et al. (1984) 
        <http://adsabs.harvard.edu/abs/1984A%26A...132..389P>`_.

        The reddening law is published as:

        .. math::
           \\frac{E(\lambda - V)}{E(B-V)} = (\\frac{A_{\lambda}}{A_V} - 1)*R_V,

        and is valid from 0.1275 - 0.32 :math:`\mu m` (B and V). 
        Values for :math:`\lambda > 0.32\mu m` (for U, J, H and K) are
        taken from `Bouchet et al. (1985) 
        <http://adsabs.harvard.edu/abs/1985A%26A...149..330B>`_.

        Interpolation is required for wavelengths in between those in 
        Prevot's table. Extrapolation is needed at shorter wavelengths 
        to cover Ly :math:`\\alpha`.

        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). 
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 2.72

        Returns:
            reddenedflux (float): Spectrum reddened according to the Prevot law.

        """
        # reddening law, E(lambda-V) / E(B-V)
        lam = np.array([1275., 1330., 1385., 1435., 1490., 1545., 1595., 1647.,
                        1700., 1755., 1810., 1860., 1910., 2000., 2115., 2220.,
                        2335., 2445., 2550., 2665., 2778., 2890., 2995., 3105.,
                        3704., 4255., 5291., 12500., 16500., 22000.])
        curve = np.array([13.54, 12.52, 11.51, 10.80, 9.84, 9.28, 9.06, 8.49,
                          8.01, 7.71, 7.17, 6.90, 6.76, 6.38, 5.85, 5.3, 4.53,
                          4.24, 3.91, 3.49, 3.15, 3.00, 2.65, 2.29, 1.81, 1.00, 
                          0.00, -2.02, -2.36, -2.47])

        # extrapolate to shorter wavelengths
        f = interp1d(lam[0:3], curve[0:3])
        atshortlam = self.extrap1d_linefit(f)(wave[wave < 1275])
        newlam = np.append(wave[wave < 1275], lam)
        newcurve = np.append(atshortlam, curve)

        # interpolate in between wavelengths
        spline = InterpolatedUnivariateSpline(newlam, newcurve)
        curve_interp = spline(wave)
    
        klam_interp = curve_interp / RV + 1

        self.prevot_klam = klam_interp * RV
        self.prevot_RV = RV
        
        return flux * 10.0**(-0.4*klam_interp*ebv)


    def seaton(self, wave, flux, ebv, RV=3.1):
        """Applies the reddening law for the MW from `Seaton (1979) 
        <http://adsabs.harvard.edu/abs/1979MNRAS.187P..73S>`_ 

        The reddening law is published as:

        .. math::
           \\frac{E(\lambda - V)}{E(B-V)} = (\\frac{A_{\lambda}}{A_V} - 1)*R_V,

        and is valid for :math:`x < 8.3 \mu m^{-1}` 
        (:math:`\lambda > 0.12 \mu m`). Extrapolation at shorter 
        wavelengths is not implemented at this time. 
        At wavelengths longer than :math:`0.365 \mu m`, the points
        from :func:`allen` are adopted to avoid :math:`k_{\lambda}` 
        being too flat in the red and near-IR regions. The reddening 
        law used here is as fit by `Fitzpatrick (1986) 
        <http://adsabs.harvard.edu/abs/1986AJ.....92.1068F>`_

        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). 
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 3.1

        Returns:
            reddenedflux (float): Spectrum reddened according to the Seaton law.

        """
        # extinction curve parameters
        c1 = -0.38
        c2 = 0.74
        c3 = 3.96
        x0 = 4.595
        g = 1.051

        # inverse wavelength, um^-1
        x = 10000. / wave
    
        # set c4=0 at all x
        curve = c1 + c2*x + c3/((x - (x0**2)/x)**2 + g**2)

        # add in factor for x >= 5.9
        w4 = np.where(x >= 5.9)

        if x[w4].shape[0] != 0:
            y = x[w4] - 5.9
            c4 = 0.26 * (0.539 * (y)**2 + 0.0564 * (y)**3)
            curve[w4] = c1 + c2*x[w4] + c3/((x[w4]-(x0**2)/x[w4])**2+g**2) + c4

        klam = curve / RV + 1

        self.seaton_klam = klam * RV
        # redward of 3330A, use Allen's extinction curve
        flux_allen = self.allen(wave, flux, ebv)
        self.seaton_klam[wave >= 3650.] = self.allen_klam[wave >= 3650.]

        self.seaton_RV = RV
        
        return flux * 10.0**(-0.4*klam*ebv)


    def fitz(self, wave, flux, ebv, RV=3.1):
        """Applies the reddening law for the LMC from `Fitzpatrick (1986) 
        <http://adsabs.harvard.edu/abs/1986AJ.....92.1068F>`_.

        The reddening law is published as:

        .. math::
           \\frac{E(\lambda - V)}{E(B-V)} = (\\frac{A_{\lambda}}{A_V} - 1)*R_V,

        and is valid for :math:`x < 8.3 \mu m^{-1}` 
        (:math:`\lambda > 0.12 \mu m`). Extrapolation at shorter 
        wavelengths is not implemented at this time. 
        At wavelengths longer than :math:`0.333 \mu m`, the points
        from :func:`allen` are adopted to avoid :math:`k_{\lambda}` 
        being too flat in the red and near-IR regions. 

        Args:
            wave (float): Wavelength array in Angstroms
            flux (float): Flux array
            ebv (float): Color excess, E(B-V). 
            RV (Optional[float]): Ratio of total selection extinction. 
                Defaults to 3.1

        Returns:
            reddenedflux (float): Spectrum reddened according to the Fitzpatrick law.
        """
        # extinction curve parameters
        c1 = -0.69
        c2 = 0.89
        c3 = 2.55
        x0 = 4.608
        g = 0.994

        # inverse wavelength, um^-1
        x = 10000. / wave
    
        # set c4=0 at all x
        curve = c1 + c2*x + c3/((x - (x0**2)/x)**2 + g**2)

        # add in factor for x >= 5.9
        w4 = np.where(x >= 5.9)
        

        if x[w4].shape[0] != 0:
            y = x[w4] - 5.9
            c4 = 0.50 * (0.539 * (y)**2 + 0.0564 * (y)**3)
            curve[w4] = c1 + c2*x[w4] + c3/((x[w4]-(x0**2)/x[w4])**2+g**2) + c4

        klam = curve / RV + 1

        self.fitz_klam = klam * RV
        # redward of 3330A, use Allen's extinction curve
        flux_allen = self.allen(wave, flux, ebv)
#        self.fitz_klam[wave >= 3330.] = self.allen_klam[wave >= 3330.]

        self.fitz_RV = RV

        return flux * 10.0**(-0.4*klam*ebv)


    def wild(self, wave, sSFR=0, mu=0, RV_st=0, RV_neb=3.707):
        """
        """
        # high or low stellar mass surface density?
        mu_0 = 3.e8  # Msol kpc^-2
        if mu < mu_0:
            #taus = -6.4 - 0.9 * sSFR
            taus = -6.3786 - 0.9353 * sSFR
        if mu > mu_0:
            #taus = -8.0 - 1.1 * sSFR  
            taus = -8.0055 - 1.0761 * sSFR

        # Wild attenuation law
        Q = 0.6*(wave/5500.)**(-1.3) + 0.4*(wave/5500.)**(-0.7)

        self.wild_klam = Q * RV_neb
        self.wild_RV = RV_neb

        return taus


    def extrap1d_linefit(self, interpolator):
        """Perform linear extrapolation using the slope fit to all points.

        Linearly extrapolate values outside the range defined for 
        interpolation by using the slope of a line fit to all points 

        Args:
            interpolator: An interpolation function, such as 
                :func:`scipy.interpolation.interp1d`

        Returns:
            A function which can also extrapolate
        """
        xs = interpolator.x
        ys = interpolator.y
    
        def fit_line(x):
            p = np.polyfit(xs, ys, 1)
            if x < xs[0]:
                return ys[0] + (x-xs[0])*p[0]
            elif x > xs[-1]:
                return ys[-1] + (x-xs[-1])*p[0]
            else:
                return interpolator(x)

        def extrapolate(xs):
            return np.array(map(fit_line, xs))

        return extrapolate


    def extrap1d_linear(self, interpolator):
        """Perform linear extrapolation using the closest points.

        Linearly extrapolate values outside the range defined for 
        interpolation by using the slope of the line connecting 
        the two closest points.

        Args:
            interpolator: An interpolation function, such as 
                :func:`scipy.interpolation.interp1d`

        Returns:
            A function which can also extrapolate
        """
        # get x,y arrays 
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                # if desired x is less than the range
                # linear interpolation:
                #   delta(x) = (xnew - x[0])
                #   ynew = y[0] + delta(x) * m
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                # if desired x is greater than range
                #   delta(x) = (xnes - x[-1])
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)

        def extrapolate(xs):
            return np.array(map(pointwise, xs))

        return extrapolate


    def plot_ext_laws(self):
        wave = np.arange(900, 20000)
        f0 = np.ones(wave.shape[0])

        for law in ['calz', 'ccm', 'allen', 'prevot', 'seaton', 'fitz']:
            getattr(self, law)(wave, f0, 1.)
        
        self.wild(wave) 

        fig = plt.figure()
        gs = GridSpec(1,1)
        gs.update(left=0.12, right=0.95, top=0.95, bottom=0.12)
        ax = fig.add_subplot(gs[0])
    
        ax.semilogx(wave, self.calz_klam, 'c', lw=1.5, label='Calzetti')
#        ax.semilogx(wave, self.ccm_klam, 'k', label='Cardelli')
        ax.semilogx(wave, self.allen_klam, 'r', lw=1.5, label='Allen')
        ax.semilogx(wave, self.prevot_klam, 'g', lw=1.5, label='Prevot')
        ax.semilogx(wave, self.seaton_klam, 'm', lw=1.5, label='Seaton')
        ax.semilogx(wave, self.fitz_klam, 'b', lw=1.5, label='Fitzpatrick')

        ax.legend(frameon=False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        
        ax.set_ylabel(r'$k(\lambda)$', fontsize=20)
        ax.set_xlabel(r'$\lambda [\AA]$', fontsize=20)
        ax.set_xlim(9e2, 2.5e4)
        ax.set_ylim(0, 20)
        plt.savefig('extlaw.pdf')


class IGMAttenuation(object):
    """Class to calculate and apply attenuation from the intergalactic medium.
    """

    def __init__(self):
        pass


    def inoue(self, wave, zz):
        """Calculates IGM attenuation following `Inoue et al. (2014) 
        <http://adsabs.harvard.edu/abs/2014MNRAS.442.1805I>`_.

        The attenuation by the intergalactic medium is taken from the 
        analytic model of Inoue et al. (2014), which is preferred over
        Madau (1995) because they inlcude more Lyman series lines, the
        contribution from damped Lyman-alpha systems, and a better 
        treatment of the Lyman continuum. 

        :func:`inoue` requires ``inoue_coeffs.dat``, which can be found
        in ``synthsrc/requiredfiles``.

        Args:
            wave (float): Array of wavelength values
            zz (float): Redshift of source, z NOT (1+z)

        Returns:
            ext (float): The absorption array, :math:`e^{-\\tau^{IGM}_{\\lambda_{obs}}}`
        """
        # observed wavelength
        lobs = wave * (1. + zz)

        # read in coefficients
        coeffs = np.genfromtxt(os.path.join(get_filesdir(),'inoue_coeffs.dat'), 
                               dtype=[('j',int), ('lam',float), 
                               ('a1_LAF',float), ('a2_LAF',float), 
                               ('a3_LAF',float), ('a1_DLA',float), 
                               ('a2_DLA',float)])
        lam = coeffs['lam']
        nlines = lam.shape[0]
    
        # Lyman series absorption
        tau_LAF_LS = np.zeros(lobs.shape[0], dtype=float)
        tau_DLA_LS = np.zeros(lobs.shape[0], dtype=float)
        for j in range(nlines):
            # check that lam_j < lam_obs < lam_j (1 + z)
            w = np.where((lobs < lam[j]*(1+zz)) & (lobs > lam[j]))
            # contribution from Lya forest
            w1 = np.where((lobs < lam[j]*(1+zz)) & 
                          (lobs > lam[j]) & (lobs < (2.2*lam[j])))
            w2 = np.where((lobs < lam[j]*(1+zz)) & (lobs > lam[j]) & 
                          ((lobs >= (2.2*lam[j])) & (lobs < (5.7*lam[j]))))
            w3 = np.where((lobs < lam[j]*(1+zz)) & (lobs > lam[j]) & 
                          (lobs >= (5.7*lam[j])))
            tau_LAF_LS[w1] += coeffs['a1_LAF'][j] * (lobs[w1] / lam[j])**1.2
            tau_LAF_LS[w2] += coeffs['a2_LAF'][j] * (lobs[w2] / lam[j])**3.7
            tau_LAF_LS[w3] += coeffs['a3_LAF'][j] * (lobs[w3] / lam[j])**5.5

            # contribution from DLAs
            w1 = np.where((lobs < lam[j]*(1+zz)) & (lobs > lam[j]) & 
                          (lobs < (3.0*lam[j])))
            w2 = np.where((lobs < lam[j]*(1+zz)) & (lobs > lam[j]) & 
                          (lobs >= (3.0*lam[j])))
            tau_DLA_LS[w1] += coeffs['a1_DLA'][j] * (lobs[w1] / lam[j])**2.0
            tau_DLA_LS[w2] += coeffs['a2_DLA'][j] * (lobs[w2] / lam[j])**3.0

        # Lyman continuum absorption
        tau_LAF_LC = np.zeros(lobs.shape[0], dtype=float)
        tau_DLA_LC = np.zeros(lobs.shape[0], dtype=float)
    
        limit = 912.
        w = np.where(lobs > limit)

        # contribution from Lya forest
        if zz < 1.2:
            w1 = np.where((lobs > limit) & (lobs < (limit*(1.+zz))))
            w2 = np.where((lobs > limit) & (lobs >= (limit*(1.+zz))))
            fact1 = lobs[w1] / limit
            tau_LAF_LC[w1] = 0.325*(fact1**1.2 - (1.+zz)**(-0.9)*fact1**2.1)
            tau_LAF_LC[w2] = 0.

        if (zz >= 1.2) & (zz < 4.7):
            w1 = np.where((lobs > limit) & (lobs < (2.2*limit)))
            w2 = np.where((lobs > limit) & (lobs >= (2.2*limit)) & 
                          (lobs < (limit*(1.+zz))))
            w3 = np.where((lobs > limit) & (lobs >= (limit*(1.+zz))))
            fact1 = lobs[w1] / limit
            fact2 = lobs[w2] / limit
            tau_LAF_LC[w1] = 2.55e-2*(1.+zz)**1.6*fact1**2.1 + \
                                    0.325*fact1**1.2 - 0.250*fact1**2.1
            tau_LAF_LC[w2] = 2.55e-2*((1.+zz)**1.6*fact2**2.1 - fact2**3.7)
            tau_LAF_LC[w3] = 0.
    
        if zz >= 4.7:
            w1 = np.where((lobs > limit) & (lobs < (2.2*limit)))
            w2 = np.where((lobs > limit) & (lobs >= (2.2*limit)) & 
                          (lobs < (5.7*limit)))
            w3 = np.where((lobs > limit) & (lobs >= (5.7*limit)) & 
                          (lobs < (limit*(1.+zz))))
            w4 = np.where((lobs > limit) & (lobs >= (limit*(1.+zz))))
            fact1 = lobs[w1] / limit
            fact2 = lobs[w2] / limit
            fact3 = lobs[w3] / limit
            tau_LAF_LC[w1] = 5.22e-4*(1.+zz)**3.4*fact1**2.1 + \
                                    0.325*fact1**1.2 - 3.14e-2*fact1**2.1
            tau_LAF_LC[w2] = 5.22e-4*(1.+zz)**3.4*fact2**2.1 + \
                                    0.218*fact2**2.1 - 2.55e-2*fact2**3.7
            tau_LAF_LC[w3] = 5.22e-4*((1.+zz)**3.4*fact3**2.1 - fact3**5.5)
            tau_LAF_LC[w4] = 0.

        # contribution from DLAs
        if zz < 2.0:
            w1 = np.where((lobs > limit) & (lobs < (limit*(1.+zz))))
            w2 = np.where((lobs > limit) & (lobs >= (limit*(1.+zz))))
            fact1 = lobs[w1] / limit
            tau_DLA_LC[w1] = 0.211*(1.+zz)**2.0 - \
                    7.66e-2*(1.+zz)**2.3*fact1**(-0.3) - 0.135*fact1**2.0
            tau_DLA_LC[w2] = 0.
    
        if zz >= 2.0:
            w1 = np.where((lobs > limit) & (lobs < 3.0*limit))
            w2 = np.where((lobs > limit) & (lobs >= (3.0*limit)) & 
                          (lobs < (limit*(1.+zz))))
            w3 = np.where((lobs > limit) & (lobs >= (limit*(1.+zz))))
            fact1 = lobs[w1] / limit
            fact2 = lobs[w2] / limit
            tau_DLA_LC[w1] = 0.634 + 4.70e-2*(1.+zz)**3.0 - \
                                    1.78e-2*(1.+zz)**3.3*fact1**(-0.3) - \
                                    0.135*fact1**2.0 - 0.291*fact1**(-0.3)
            tau_DLA_LC[w2] = 4.70e-2*(1.+zz)**3.0 - \
                    1.78e-2*(1.+zz)**3.3*fact2**(-0.3) - 2.92e-2*fact2**3.0
            tau_DLA_LC[w3] = 0.

        tau = tau_LAF_LS + tau_DLA_LS + tau_LAF_LC + tau_DLA_LC
        
        ext = np.exp(-tau)

        return ext

