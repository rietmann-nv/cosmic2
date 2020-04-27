#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @author: smarchesini
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Code specific to the FCCD detector, including scrambling/descrambling, masking, bad-pixels, etc..."""
import numpy as np
import scipy.special as special
import scipy.signal as signal
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm

# output
frameWidth=256, 
frameHeight=256,


## ALS fccd properties:
# hnu: single photon counts (ADU):
E = 1300 #eV
hnu = E* 1.15e-2 #
nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux

#nrows = 970 #number of rows
nrows = 520
gap = 33
nrows1=nrows-gap # good rows


pixelsize = 30e-6 # pixel size in meters

#hnu=15, 
#nbcol=12, 
#nb=16, 
#nrows=970, 
width=1940 
height=1152
#height=1152//2

                 
croppedWidth=960, 
croppedHeight=960,

rotate=True



# Derived detector dimensions
nbmux  = nb * nmux # Nr. of ADC channels
ncols  = nbcol * nb * nmux # Nr. of columns
npcols = nbpcol * nb * nmux # Nr. of physical columns

# Initialize rawdata/bgstd with zeros
rawdata = np.zeros((height * width // nbmux, nbmux))
bgstd   = np.zeros((height * width // nbmux, nbmux))

# Clock
tadc   = np.arange(1, nbcol * nrows + 1).astype(np.float).reshape((1, nbcol * nrows))

# Index mapping to de-multiplex
q1 = np.arange(nbmux, dtype=np.int16)
q2 = ((nbmux - q1 - 1) % nmux) * nb
q3 = 4 * np.floor(q1 / (nmux * 4))
q4 = np.floor((nbmux - q1 - 1) / nmux) % 4
q = ((q2 + nb - q3) - q4).astype(np.int16) - 1
iq = np.arange(q.shape[0])
iq[q] = np.arange(q.shape[0])
gii = (np.arange(npcols/2) + np.floor((np.arange(npcols/2))/nbpcol)*2).astype(np.int)
aii = (np.arange(ncols/2)  + np.floor((np.arange(ncols/2))/(nbmux*nbcol))*2).astype(np.int)



# Bad pixel mask
mskt = (tadc < ((nrows-1) * nbcol+1))&(tadc>((2)*nbcol))
#mskt[0,7692] = False # This is a hack to remove some artefacts (there might be a better way to do this)
#mskt[0,7694:7704] = False # This is a hack to remove some artefacts (there might be a better way to do this)
#mskt[0,7705:7716] = False # This is a hack to remove some artefacts (there might be a better way to do this)
mskadc = np.ones(nbmux).astype(np.bool)
#mskadc[82:96] = False
#mskadc[104]   = False
#mskadc[191]   = False
#mskadc[96]    = False


# Vandermonde filer 
t1  = tadc - np.min(tadc)
t1  = t1/np.max(t1) - 0.5
tt1 = 0.5 - t1
toffset = 0.6
tt2 = (tt1 - toffset) * (tt1>toffset) * special.erf((tt1-toffset)*2)
#vanderX = np.vstack([np.ones_like(t1), tt1, tt1**2, tt2, tt2**2])
vanderX = np.vstack([np.ones_like(t1), tt1, tt1**2])
#vanderX = np.vstack([np.ones_like(t1)])
msktoffset1 = (tadc>(2*nbcol)) & (((tadc-2) % nbcol)>9) #this is corret (mask only every 10th block)
#msktoffset  = msktoffset1 & (tadc < ((nrows - 100)*nbcol))
msktoffset  = msktoffset1 & (tadc > (400*nbcol)) & (tadc < ((nrows - 80)*nbcol))
VS = vanderX[:,msktoffset[0]]
vanderfilterX = np.dot(vanderX.T, np.linalg.lstsq(np.dot(VS,VS.T), VS)[0])
#vanderfilterX = np.dot(msktoffset.T, np.linalg.lstsq(np.dot(VS,VS.T), VS)[0])
vanderfilter = lambda data: np.dot(vanderfilterX, data[msktoffset[0],:])

# Non-physical pixel
nmskadc = ~mskadc
nmskadc=np.tile(nmskadc,(12,1)).T.ravel()
nmskadc_list = np.where(nmskadc)[0]
a = (msktoffset1 & (tadc > nbcol*nrows*3/4)).T 
mskbgf = lambda data: msktf( ( (data>-hnu*1) & (data<hnu*1)) | a)
bgcolf = lambda data: np.sum(rowXclock(data * mskbgf(data)),axis=0) / np.sum(rowXclock(mskbgf(data)),axis=0)

# Fourier matching
filter  = np.ones(5)
mskbg2f = lambda data: msktf( (data>-hnu*3) & (data<hnu) )
powerspec = lambda data: np.sum(np.abs(np.fft.fft(mskbg2f(data)*data)),axis=1)
mskFclock = lambda pspec: (np.convolve(np.single((pspec-np.mean(pspec)>(pspec[0])/20.) ), filter, 'same') > 0).reshape((pspec.shape[0],1))

# CCD Mask
filter2 = np.ones((2,2))
mskccdf = lambda data: (signal.convolve2d(np.single(data > hnu),filter2,'same')>0) * data

# Pixelsize
pixelsize = pixelsize

# Dimenstions for Cropping 
Mx, My = (croppedWidth,croppedHeight)
mx, my = (frameWidth,frameHeight)

# # Semi-transparent beamtop
# beamstop_radius = beamstop_radius
# beamstop_transmission = beamstop_transmission
# beamstop_xshift = beamstop_xshift
# beamstop_yshift = beamstop_yshift

# # A circular mask for smoothing
# x = np.arange(mx) - mx/2 + 1
# y = np.arange(my) - my/2 + 1
# xx,yy = np.meshgrid(x,y)
# r2  = (xx**2 + yy**2)
# msksmooth = (special.erf( ((mx/2)*0.99 - 1.4*np.sqrt(r2)) / 20) + 1) / 2

# # A filter for attenuation/deattenation of beamstop
# x = np.arange(Mx) - Mx/2 + beamstop_xshift
# y = np.arange(My*2) - My + beamstop_yshift
# xx,yy = np.meshgrid(x,y)
# r2  = (xx**2 + yy**2)
# filter_beamstop = (r2>(beamstop_radius/pixelsize)**2) + (r2 <= (beamstop_radius/pixelsize)**2)*beamstop_transmission

# Preproc options
rotate = rotate


def blocksXtif1(data):
    """Translates `ccd` format to `row` format."""
    return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2,:],2),data[:nrows1,:]),axis=1)

def clockXblocks1(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows1, nbcol, nbmux), order='F'),[1,0,2]), (nrows1 * nbcol, nbmux), order='F')


def blocksXtif(data):
    """Translates `ccd` format to `row` format."""
    out = np.zeros((nrows,ncols))
    out[:,:ncols//2] = np.rot90(data[nrows:2*(nrows),:],2)
    out[:,ncols//2:] = data[:nrows,:]
    return out
    #return np.hstack([np.rot90(data[nrows:,:],2), data[:nrows,:]])


def clockXblocks(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows, nbcol, nbmux), order='F'),[1,0,2]), (nrows * nbcol, nbmux), order='F')



def msktf(data):
    """Returns masked data"""
    return mskadc*(data*mskt.transpose())

def adcmask(bgstd, adcthreshold=50):
    """Masking wrong adc values."""
    bgstd[:bgstd.shape[0] / nbmux,:] = bgstd.reshape((bgstd.shape[0] / nbmux, nbmux))
    adcstd = np.mean(clockXraw(bgstd), axis=0)
    badadc = adcstd < (np.median(adcstd) + adcthreshold)
    return badadc

def rowXclock(data):
    """Translates `clock` format to `row` format."""
    return np.reshape(np.transpose(np.reshape(data,(nbcol, nrows, nbmux), order='F'), [1,0,2]), (nrows, nbmux*nbcol), order='F')

def ccdXrow(data):
    """Translates `row` format to `ccd` format."""
    #return np.vstack([data[5:5+960,ncols/2+gii+1], np.rot90(data[5:5+960,gii+1], 2)])
    off = 10 
    return np.vstack([data[:-off,ncols/2+gii+1], np.rot90(data[:-off,gii+1], 2)])

def tifXrow(data):
    """Translates `row` format to `tif` format."""
    tif = np.vstack([data[:,ncols//2+aii], np.rot90(data[:,aii], 2)])
    return tif

def rowXccd(data):
    """Translates `ccd` format to `row` format."""
    out = np.zeros((nrows,ncols))
    off = 10   # This number should be deducted from the input, for example a discrepancey between `width` and `nrows`
    #out[off/2:-off/2,gii] = np.rot90(data[nrows-off:2*(nrows-off),:],2)
    out[:-off,gii] = np.rot90(data[nrows-off:2*(nrows-off),:],2)
    #out[off/2:-off/2,ncols/2 + gii] = data[:nrows-off,:]
    out[:-off,ncols/2 + gii] = data[:nrows-off,:]
    return out
    #return np.hstack([np.rot90(data[nrows:,:],2), data[:nrows,:]])

def clockXrow(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows, nbcol, nbmux), order='F'),[1,0,2]), (nrows * nbcol, nbmux), order='F')

def rowXtif(data):
    """Translates `tif` format to `row` format."""
    return np.reshape(np.roll(np.reshape(rowXccd(data), (nrows, nbcol, nbmux)), -1, axis=2),(nrows, nbcol*nbmux))

def clockXtif(data):
    """Translates `tif` format to `clock` format."""
    return clockXrow(rowXtif(data))

def clockXraw(data):
    """Translates `raw` format to `clock` format. This was the slowest operation in the descrambling process"""
    #return data.reshape((nbcol * nrows, nb*nmux)).transpose()[q,:].transpose()
    #return data.reshape((nbcol * nrows, nb*nmux))
    #return data.reshape((nbcol * nrows, nb*nmux), order='F')[:,q]
    return data.reshape((nbcol * nrows, nb*nmux))[:,q]
    
def rawXclock(data):
    """Translates `clock` format to `raw` format."""
    return data.reshape((nbcol * nrows, nb*nmux), order='F')[:,iq]

def cropimg(img ,s, dx=0, dy=0):
    """Returns cropped image."""
    c0 = np.floor((img.shape[0] - s[0])/2 + dx)
    c1 = np.floor((img.shape[1] - s[1])/2 + dy)
    return img[c0:c0+s[0],c1:c1+s[1]]

def cropF(self,img):
    """Returns image cropped to (croppedWidth, croppedHeight)."""
    return cropimg(img,(My, Mx))

def cropframe(self,img):
    """Returns image cropped to (frameWidth, frameHeight)."""
    return cropimg(img,(my, mx), dx=beamstop_xshift, dy=beamstop_yshift)

def cropframe_smooth(self,img):
    """Returns image smoothed and cropped to (frameWidth, frameHeight)."""
    return cropframe(img)*msksmooth

def downsample(frame):
    """Returns downsampled frame."""
    return np.abs(np.fft.fft2(cropframe_smooth(np.fft.fftshift(np.fft.ifft2(frame)))))

def lowpass(frame):
    """Returns filtered frame."""
    return downsample(cropF(frame))

def deattenuate(frame):
    """Returns frame with deattenuated beamstop area."""
    return frame / filter_beamstop

def attenuate(frame):
    """Returns frame with attenuated beamstop area."""
    return frame * filter_beamstop

def tif2raw(img):
    """Returns scrambled data, translating `tif` format to `raw` format."""
    clockdata = clockXtif(img)
    return rawXclock(clockdata)

def descramble(rawdata, badadc_mask=None, mask=False):
    """Returns descrambled data, translating `raw` format to `clock` format."""
    rawdata[:rawdata.shape[0] / nbmux,:] = rawdata.reshape((rawdata.shape[0] / nbmux, nbmux))
    if mask:
        clockdata = msktf(clockXraw(rawdata.reshape((height, width))))
    else:
        clockdata = clockXraw(rawdata.reshape((height, width)))
    if badadc_mask is not None:
        clockdata = clockdata*badadc_mask
    return clockdata

def scramble(data):
    """Returns scrambled data, translating `ccd` format ro `raw` format."""
    clockdata = clockXrow(rowXccd(data))
    return rawXclock(clockdata)

def preprocessing(data):
    """Returns clean data after removing of negative pixels, filtering (vandermonde), removing
    of non-physical pixels and more filtering (fourier)."""

    # 1. Removing negative pixels
    data[data < (-hnu*5)] = 0.

    # 2. Remove polynomials (vandermonde filter)
    data = data - vanderfilter(data)
    
    # 3. Remove non-physical pixels
    bgcol = bgcolf(data)
    bgcol[nmskadc_list] = 0
    bgcol = clockXrow(np.vstack([np.zeros((2,ncols)),np.tile(bgcol,(nrows-37,1)),np.zeros((35,ncols))]))
    data = msktf(data - bgcol)
    
    # 4. Fourier filter
    bgfilt = np.zeros_like(data)
    for jj in range(2):
        bgfilt = mskbgf(data) * data + (~mskbgf(data)) * bgfilt
        bgfilt = np.real(np.fft.ifft(np.fft.fft(bgfilt) * mskFclock(powerspec(data))))
    data = data - bgfilt

    # 5. Downsample and low pass filter
    data = lowpass(deattenuate(assemble(data)))

    # 6. Rotate the image by 90 degress counter clockwise
    if rotate:
        data = np.rot90(data, -1)
    
    return data
        
def assemble(clockdata):
    """Returns assembled data, translating 'clock' format to 'ccd' format."""
    return mskccdf(ccdXrow(rowXclock(clockdata)))

def assemble_nomask(clockdata):
    """Returns assembled data, translating 'clock' format to 'ccd' format."""
    return ccdXrow(rowXclock(clockdata))

def assemble2(clockdata):
    """Returns assembled data, translating 'clock' format to 'ccd' format."""
    return tifXrow(rowXclock(clockdata))
    
    


