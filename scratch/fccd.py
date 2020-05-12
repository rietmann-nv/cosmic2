#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

#import scipy.constants
 
# fccd readout
nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

# physical properties
#hnu = E* 1.15e-2 # single photon in ADU (approx)
ccd_pixel=30e-6  # 30 microns

# shifting/cropping frames
nrcols=484

ngcols=480
heigth=ngcols*2
width = heigth

nrows = 520
#gap = 33
gap = 34
nrows1=nrows-gap # good rows


# coordinates in the clean image
xx=np.linspace(-1,1,2*ngcols)



def clockXblocks1(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows1, nbcol, nbmux), order='F'),[1,0,2]), (nrows1 * nbcol, nbmux), order='F')

def blocksXtif1(data):
    """Translates `ccd` format to `row` format."""
    #return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2,:],2),data[:nrows1,:]),axis=1)
    #return np.concatenate((np.rot90(data[nrows1+1+gap*2:2*(nrows1)+gap*2-1,:],2),data[2:nrows1,:]),axis=1)
    #return np.concatenate((np.rot90(data[nrows1+1+gap*2:2*(nrows1)+gap*2-1,:],2),data[1:nrows1-1,:]),axis=1)
    return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2-2,:],2),data[1:nrows1-1,:]),axis=1)

def bblocksXtif1(data): # stack the blocks
    return np.reshape(blocksXtif1(data),(nrcols,nbmux,nbcol))
    #return np.reshape(blocksXtif1(data),(nbmux,nrcols,nbcol))


def tif1Xbblocks(data): # tif from stacked blocks 
    return np.reshape(data[:,:,1:nbcol-1],(nrcols,nbmux*nbpcol))

def imgXtif1(data): # final image from tif1
    #return np.concatenate((data[6:486,0:960],np.rot90(data[6:486,960:],2)))
    #return np.concatenate((data[5:485,0:960],np.rot90(data[5:485,960:],2)))
    return np.concatenate((data[4:484,0:960],np.rot90(data[4:484,960:],2)))


        
# combine double exposure
def combine(data0, data1, t12, thres=3e3):
    msk=data0<thres
    return (t12+1)*(data0*msk+data1)/(t12*msk+1)

######################3
# denoise bblocks
bpts=60//2
gg=np.exp(-(np.arange(-bpts//2,bpts//2)/(bpts/4))**2)
gg/=np.sum(gg)
#gg=np.reshape(gg,(bpts,1))

def conv2d(data,filt):
    data_s=np.empty(np.shape(data))
    nr=data.shape[1]

    for r in range(nr):
        data_s[:,r] = np.convolve(data[:,r], gg, 'same')
    return data_s

def filter_bblocks(data):
    #yy=np.reshape(data[:,:,0],(nrcols,nbmux))
    # vertical stripes
    yy=np.reshape(data[:,:,11],(nrcols,192))
    # clip and smooth
    bkgthr=5 # background threshold
    yy_s=conv2d(np.clip(yy,-bkgthr,bkgthr),gg)
    yy_s=np.reshape(yy_s,(nrcols,nbmux,1))

    data_out = data-yy_s
    #data_out = data#-yy_s
    ###yy_avg=np.reshape(np.average(np.clip(bblocksXtif1(data_out)[1:11,:,:],0,2*bkgthr),axis=0),(1,192,12))
    yy_avg=np.reshape(np.average(np.clip(data_out[1:10,:,:],0,2*bkgthr),axis=0),(1,192,12))
    data_out -= yy_avg
    data_out *= data_out>7
    return data_out#-yy_s-yy_avg

######################3

def imgXraw_nofilter(data): # combine operations
    return imgXtif1(tif1Xbblocks(bblocksXtif1(data)))
  #  return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))


def imgXraw(data): # combine operations
    #return imgXraw_nofilter(data)
    return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))



'''

#########################
# from metadata
# Energy (converted to keV)
# E= fid['entry_1/instrument_1/source_1/energy'][...]*1/scipy.constants.elementary_charge
# get the width from the desired resolution

#if final_res is not None:
#    img_width= heigth/(ccd_dist*wavelength/(ccd_pixel*heigth)/final_res) # cropped width of the raw clean frames

smooth_factor=10
filter_width=frame_pixels
#xx1c = np.fft.ifftshift(xx1)
xx1c = np.fft.ifftshift(xx1)*width/img_width

sth1= -np.fft.fftshift(np.arctan((np.abs(xx1c)-filter_width)/smooth_factor))/np.pi*2
sth1*=sth1>0
sth1*=sth1
sth1 = sth1*sth1.T

rr1c = np.sqrt(np.fft.ifftshift(xx1**2+(xx1.T)**2))*width/img_width

sth=-np.fft.fftshift(np.arctan((rr1c-filter_width)/smooth_factor))/np.pi*2
sth*=sth*(sth>0)

#sth = np.fft.fftshift(np.exp(-(rr1c/filter_width/2)**2))

sth = sth1

ccdw = ngcols*2

"""
def shift_img0(img,shift,flt=sth):
    #img_max=np.max(img)
    #soft_filt=np.arctan((img-img_max*.1)/(img_max*.1))*2/np.pi+1

    Fimg=np.fft.fft2(img)
    #Fimg=np.fft.fft2(img*(1-soft_filt))
    Fimg*=np.exp(1j*2*np.pi*shift[0]*xx1/xdim)
    Fimg*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    Fimg=np.real(np.fft.ifft2(Fimg))

    #Fimg1=np.fft.fft2(img*soft_filt)
    #Fimg1*=np.exp(1j*2*np.pi*shift[0]*xx1/xdim)*flt
    #Fimg1*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    #Fimg1=np.real(np.fft.ifft2(Fimg1))
    #Fimg+=Fimg1
    

    
    #set to 0 the wrap-around indices
    Fimg[:, np.int(np.min([ccdw,np.floor(ccdw-shift[0])])):ccdw]=0
    Fimg[np.int(np.min([ccdw,np.floor(ccdw-shift[1])])):ccdw,:]=0
    Fimg[np.int(np.max([0,-shift[0]])):np.max([0,-np.int(np.ceil(shift[0]))]),:]=0
    Fimg[:,np.int(np.max([0,-shift[1]])):np.max([0,-np.int(np.ceil(shift[1]))])]=0
    
    return Fimg

def shift_img(img,shift,flt=sth):
    img_max=np.max(img)
    #soft_filt=np.arctan((img-img_max*.1)/(img_max*.1))*2/np.pi+1
    soft_filt=np.arctan((img-img_max*.1)/(img_max*1e-2))/np.pi+.5
    
    #Fimg=np.fft.fft2(img)
    Fimg=np.fft.fft2(img*(1-soft_filt))
    Fimg*=np.exp(1j*2*np.pi*shift[0]*xx1/xdim)*flt
    Fimg*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    Fimg=np.real(np.fft.ifft2(Fimg))

    Fimg1=np.fft.fft2(img*soft_filt)
    Fimg1*=np.exp(1j*2*np.pi*shift[0]*xx1/xdim)
    Fimg1*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    Fimg1=np.real(np.fft.ifft2(Fimg1))
    
    Fimg+=Fimg1
    

    
    #set to 0 the wrap-around indices
    Fimg[:, np.int(np.min([ccdw,np.floor(ccdw-shift[0])])):ccdw]=0
    Fimg[np.int(np.min([ccdw,np.floor(ccdw-shift[1])])):ccdw,:]=0
    Fimg[np.int(np.max([0,-shift[0]])):np.max([0,-np.int(np.ceil(shift[0]))]),:]=0
    Fimg[:,np.int(np.max([0,-shift[1]])):np.max([0,-np.int(np.ceil(shift[1]))])]=0
    
    return Fimg
"""

def filter_img(img,flt=sth):
    #img_max=np.max(img)
    #soft_filt=np.arctan((img-img_max*.1)/(img_max*.1))*2/np.pi+1

    Fimg=np.fft.fft2(img)
    #Fimg=np.fft.fft2(img*(1-soft_filt))
    Fimg*=flt
    #Fimg*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    Fimg=np.abs(np.fft.ifft2(Fimg))
    Fimg*=Fimg>0
    
    return Fimg

import scipy.signal
box_width=np.max([np.int(np.floor(img_width/frame_pixels)),1])

bbox=np.ones((box_width,box_width))

def filter_img0(img):
    return scipy.signal.convolve2d(img,bbox,mode='same',boundary='fill')
    
    
def filter_img1(img,flt=sth):
    img_max=np.max(img)
    #soft_filt=np.arctan((img-img_max*.1)/(img_max*1e-2))/np.pi+.5
    soft_filt=np.arctan((img-img_max*.005)/(img_max*1e-3))/np.pi+.5
    
    #Fimg=np.fft.fft2(img)
    Fimg=np.fft.fft2(img*(1-soft_filt))
    Fimg*=flt
    #Fimg*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)#*flt.T
    Fimg=np.abs(np.fft.ifft2(Fimg))
    
    Fimg+=img*soft_filt

    
    return Fimg


from scipy import interpolate
coord=np.arange(-frame_pixels//2,frame_pixels//2)/frame_pixels*img_width+xdim//2
def rescale(img):
    return (interpolate.interp2d(xx, xx, img, fill_value=0)(coord,coord)).T

 

# characterization of the dark field 
# get the background
bkg=np.array(fid['entry_1/data_1/dark_data'])

# split the average from 2 exposures:
bkg_avg0=np.average(bkg[0::2],axis=0)
bkg_avg1=np.average(bkg[1::2],axis=0)

## get one frame to compute center
rdata = fid['entry_1/data_1/raw_data']
n_frames = rdata.shape[0]//2 # number of frames (1/2 for double exposure)

# split short and long exposures
ii=2500//2+25 # middle frame, we could use the first
rdata0=rdata[ii*2]-bkg_avg0
rdata1=rdata[ii*2+1]-bkg_avg1

img0=combine(imgXraw(rdata0),imgXraw(rdata1))

# we need a shift, we take it from the first frame:
#com = np.round(center_of_mass(img0*(img0>0))-xdim//2)
com = center_of_mass(img0*(img0>0))-xdim//2
com = np.round(com)

def shift_rescale(img):
    return (interpolate.interp2d(xx, xx, img, fill_value=0)(coord+com[1],coord+com[0])).T


# incorporate the shift
#def center_img(img):
#    return shift_img(img0, com)


# plot an image

if Figon:
    img2 = filter_img(img0)
    #img3 = rescale(img2)
    img3 = shift_rescale(img2)
    img3 = img3*(img3>0) # positive



# modify pixel size
# pixel size is rescaled
x_pixel_size=ccd_pixel*img_width/frame_pixels
#####################




figure = None
for ii in np.arange(n_frames):

    img0 = combine(imgXraw(rdata[ii*2]-bkg_avg0),imgXraw(rdata[ii*2+1]-bkg_avg1))

    #img2 = filter_img1(img0)
    img2 = filter_img(img0)
    #img2 = img0
    # block out the center pixels
    #bwidth=5
    #img2[ngcols-bwidth:ngcols+bwidth+1,ngcols-bwidth:ngcols+bwidth+1]=0
    
    #img3 = rescale(img2)
    img3 = shift_rescale(img2)
    # force it to be positive
    img3 = img3*(img3>0)
'''
