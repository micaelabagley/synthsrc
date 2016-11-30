#! /usr/bin/env python
import os
import re
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, Column
import matplotlib.pyplot as plt


class RandomCatalog(object):
    """Class to create random realizations of the synthetic catalog.

    Args:
        phottype (str):
        limtype (Optional[str]):
        wispcat (Optional[str]):
        makeplot (Optional[str]):
        
    """
    
    def __init__(self, outfits, sig, nrealz, randcat, phottype, limtype, plotsdir, wispcat, makeplot=True):
        self.plots = plotsdir
        self.outcat = randcat
        self.nrealz = nrealz

        self.sig814 = sig

        # for calculating the errors within which to randomize photometry
        self.phottype = phottype
        self.limtype = limtype
        if wispcat is None:
            wispcat = '/data/highzgal/bagley/wisps/wispcatalogs/full_catalog_aper.fits'
        self.wispcat = fits.getdata(wispcat)
        self.magbins = np.arange(15, 30, 0.5)
        self.bc = 0.5 * (self.magbins[1:] + self.magbins[:-1])
        self.nbins = self.bc.shape[0] 

        self.makeplot = makeplot
        self.filts = ['F606W', 'F814W', 'F110W', 'F160W']
        self.colors = ['b', 'g', '#ffa500', 'r']
 
        # synthsrc synthetic catalog
        self.synthcat = Table.read(outfits)

        nsrc = self.synthcat['z'].shape[0]
        print nsrc
        self.synthcat.add_column(Column(data=np.ones(nsrc, dtype=int),
                                 name='flag814'))
        self.synthcat.add_column(Column(data=np.ones(nsrc, dtype=int),
                                 name='flag606'))

        self.maglimits, self.magdict, self.zps = self.get_maglimits()
 
        self.randomize_catalog()


    def get_maglimits(self):
        """Gets magnitude limits in WFC3 filters for the WISP survey.

        Magnitude limits are measured by dropping random circular 
        apertures on the background in each WISP image and fitting
        Gaussians to the distributions of fluxes. 
        These values are currently hard-coded. 
        
        ======  =========   ======  ======  ======  ======
        Filter  zeropoint   min     max     median  mean
        ======  =========   ======  ======  ======  ======
        F606W   26.0691     0.1078  0.1916  0.1491  0.1463
        \       \           28.488  27.863  28.135  28.156
        F814W   25.0985     0.0710  0.1260  0.0935  0.0945
        F110W   26.8223     0.1786  0.3666  0.2518  0.2553
        F160W   25.9463     0.1995  0.3266  0.2570  0.2571
        
        Returns:
            (tuple): tuple containing:
    
                ml (dict):
                magdict (dict):
                zps (dict): 
        """
        F606W = {'min':0.1078, 'max':0.1916, 'med':0.1491, 'mean':0.1463}
        F814W = {'min':0.0710, 'max':0.1260, 'med':0.0935, 'mean':0.0945}
        F110W = {'min':0.1786, 'max':0.3666, 'med':0.2518, 'mean':0.2553}
        F160W = {'min':0.1995, 'max':0.3266, 'med':0.2570, 'mean':0.2571}

        m606 = -2.5 * np.log10(F606W[self.limtype]) + 26.0691
        m814 = -2.5 * np.log10(self.sig814*F814W[self.limtype]) + 25.0985
        m110 = -2.5 * np.log10(F110W[self.limtype]) + 26.8223
        m160 = -2.5 * np.log10(F160W[self.limtype]) + 25.9463

        ml = {'F606W':m606, 'F814W':m814, 'F110W':m110, 'F160W':m160}
        magdict = {'F606W':F606W, 'F814W':F814W, 'F110W':F110W, 'F160W':F160W}
        zps = {'F606W':26.0691,'F814W':25.0985,'F110W':26.8223,'F160W':25.9463}
        return ml, magdict, zps


    def reject_outliers(self, data, m=3):
        """Reject outliers from a data set using sigma clipping.

        Args:
            data (float): Data array
            m (Optional[int]): 

        Returns:
            data (float): Data array clipped 
        """
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0
        return data[s<m]


    def get_catalog_magnitudes(self, filt):
        """Read in photometry and uncertainties from WISP catalog. 

        Args:
            filt (float)

        Returns:
            (tuple): Tuple containing:

                mag (float):
            
                emag (float):
        """
        mag = self.wispcat['MAG_%s_%s'%(self.phottype,filt)]
        emag = self.wispcat['MAGERR_%s_%s'%(self.phottype,filt)]
        # missing photometry (i.e. no F606W for field) masked by nans
        w = np.isfinite(mag) & np.isfinite(emag)
        mag = mag[w]
        emag = emag[w]        
        # remove sources that are undetected (m=99)
        #        or off image / on chip gap (m=-99)
        #        or with large errors
        wgood = np.where((mag != -99) & (mag != 99) & (emag < 5))

        return mag[wgood], emag[wgood]
    

    def bin_errors(self, filt, c=None, ax=None, makeplot=False):
        """
        """
        mederr = np.zeros(self.nbins, dtype=float)
        std = np.zeros(self.nbins, dtype=float)
        mag, emag = self.get_catalog_magnitudes(filt)
        inds = np.digitize(mag, self.magbins)
        
        for i in range(self.nbins):
            errors = emag[inds == (i+1)]
            if errors.shape[0] > 5:
                errs = self.reject_outliers(errors, m=5)
                mederr[i] = np.median(errs)
                std[i] = np.std(errs)

        # replace bins with too few sources with median and std 
        # of neighboring bins
        nzero = np.where(mederr != 0)
        std[(self.bc < 20) & (mederr == 0)] = std[nzero[0][0]]
        mederr[(self.bc < 20) & (mederr == 0)] = mederr[nzero[0][0]]

        # cutoff at limiting magnitude of filter and replace 
        mlim = self.maglimits[filt]
        wlim = np.where(self.bc <= mlim)
        std[(self.bc > mlim)] = std[wlim[0][-1]]
        mederr[(self.bc > mlim)] = mederr[wlim[0][-1]]
     
        if makeplot:
            self.plot_errors(mag, emag, mederr, std, c, filt, ax)
        
        return mederr, std

    
    def randomize_catalog(self):
        """
        """
        d606 = self.magdict['F606W']
        d814 = self.magdict['F814W']
        d160 = self.magdict['F160W']
        min606 = -2.5 * np.log10(d606['min']) + self.zps['F606W']
        max606 = -2.5 * np.log10(d606['max']) + self.zps['F606W']
        min814 = -2.5 * np.log10(self.sig814*d814['min']) + self.zps['F814W']
        max814 = -2.5 * np.log10(self.sig814*d814['max']) + self.zps['F814W']
        min160 = -2.5 * np.log10(d160['min']) + self.zps['F160W']
        max160 = -2.5 * np.log10(d160['max']) + self.zps['F160W']

        # remove sources that are below the F110W detection limit
        # sources that are far too faint can scatter up to 1 mag back 
        # into catalog
        w = np.where(self.synthcat['m110'] > self.maglimits['F110W'])
        self.synthcat.remove_rows(w[0])

        if self.makeplot:
            fig,ax = plt.subplots(1,1)

        for i in range(self.nrealz):
            print i
            t0 = self.synthcat.copy()

            j = 0
            for j,filt in enumerate(self.filts):
                f = re.search('\d+',filt).group(0)
                catmag = self.synthcat['m%s'%f]
                
                inds = np.digitize(catmag, self.magbins)
                
                if (self.makeplot) & (i == 0):
                    mederr,std = self.bin_errors(filt, c=self.colors[j], ax=ax,
                                                 makeplot=True)
                else:
                    mederr,std = self.bin_errors(filt)
                
                # indices of 0 or >nbins are for sources that are outside
                # the madnitude bin range. use the median error of the 
                # closest bin
                inds[inds == 0] = 1
                inds[inds > self.nbins] = self.nbins
                meds = np.array([mederr[x-1] for x in inds])

                new = np.random.normal(catmag, meds)

                # randomize the limits a bit so they're not all at
                # the median value
                if filt == 'F814W':
                    # detection in F814 depends on sig (1, 1.5, 2, ...)
                    w = np.where(new > self.maglimits['F814W'])
                    new[w] = np.random.uniform(min814,max814,catmag[w].shape[0])
                    self.synthcat['flag814'][w] = 0
                if filt == 'F606W':
                    # detection in F606 must always be 1 sigma
                    w = np.where(new > self.maglimits['F606W'])
                    new[w] = np.random.uniform(min606,max606,catmag[w].shape[0])
                    self.synthcat['flag606'][w] = 0
                
                if filt == 'F160W':
                    w = np.where(new > self.maglimits['F160W'])
                    new[w] = np.random.uniform(min160,max160,catmag[w].shape[0])

                t0['m%s'%f] = new
                
            if i != 0:
                t = vstack([t, t0])
            else:
                t = t0
        t.write(self.outcat, format='fits')

        if self.makeplot:
            ax.legend(scatterpoints=1, loc=2)
            ax.set_xlim(15,29)
            ax.set_ylim(-0.05, 1.5)
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('Error')
            plt.tight_layout()
            fig.savefig(os.path.join(self.plots, 'photscatter.pdf'))
       

    def plot_errors(self, mag, emag, mederr, std, c, filt, ax):
        """
        """
        ax.scatter(mag, emag, marker='.', edgecolor='none', alpha=0.1,
                   color=c)
        ax.scatter(self.bc, mederr, marker='o', s=40, edgecolor=c, 
                   color='w', label=filt)
        ax.fill_between(self.bc, mederr-std, mederr+std, edgecolor='none',
                        alpha=0.2, color=c)


if __name__ == '__main__':
    cat = RandomCatalog('APER', 3)
