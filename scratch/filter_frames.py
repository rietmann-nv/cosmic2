#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate

def init(width, frame_pixels, img_width):
    
        
    #print('width'+width)
    # Center of mass (COM):
    xdim=width
    xx=np.reshape(np.arange(xdim),(xdim,1))
    #xx1=xx-(xx+1>xdim//2)*xdim
    
    def center_of_mass(img):
        return np.array([np.sum(img*xx)/np.sum(img),  np.sum(img*xx.T)/np.sum(img)])
    
    # filters
    
    import scipy.signal
    box_width=np.max([np.int(np.floor(img_width/frame_pixels)),1])
    
    bbox=np.ones((box_width,box_width))
    
    def filter_img0(img):
        return scipy.signal.convolve2d(img,bbox,mode='same',boundary='fill')
        
    
    
    coord=np.arange(-frame_pixels//2,frame_pixels//2)/frame_pixels*img_width+xdim//2
    
    
    def shift_rescale(img,com):
        img_out=(interpolate.interp2d(xx, xx, img, fill_value=0)(coord+com[1],coord+com[0])).T
        img_out*=(img_out>0)
        return img_out
     
    return center_of_mass, filter_img0, shift_rescale
    #ccdw = ngcols*2

import scipy.constants
hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge

#
# get the width from the desired resolution
# width=img0.shape[0]

def resolution2frame_width(ccd_dist,E,ccd_pixel,width,final_res):
    wavelength = hc/E
    padded_frame_width = width**2*ccd_pixel*final_res/(ccd_dist*wavelength)

    return padded_frame_width # cropped width of the raw clean frames


"""
nothing here

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

"""


"""
#def rescale(img):
#    return (interpolate.interp2d(xx, xx, img, fill_value=0)(coord,coord)).T

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
"""



# incorporate the shift
#def center_img(img):
#    return shift_img(img0, com)


# plot an image

#img2 = filter_img(img0)
#img3 = rescale(img2)
#img3 = shift_rescale(img2)
#img3 = img3*(img3>0) # positive


