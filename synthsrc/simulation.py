import os
from glob import glob
from datetime import datetime

import synthsrc

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

