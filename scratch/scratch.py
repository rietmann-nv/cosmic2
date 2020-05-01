#!/usr/bin/env python3
# -*- coding: utf-8 -*-

E = 1300 #eV

frame_pixels=256 # final pixels 
img_width= 600 # cropped width of the raw clean frames

# fccd readout
nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

# physical properties
hnu = E* 1.15e-2 # single photon in ADU (approx)
ccd_pixel=30e-6  # 30 microns

t12 = 5 # time ratio between long and short exposure


# shifting/cropping frames
#nrcols=487
nrcols=485

ngcols=480
nrows = 520
gap = 33
nrows1=nrows-gap # good rows

# pixel size is rescaled
x_pixel_size=ccd_pixel*img_width/frame_pixels



import numpy as np

# coordinates in the clean image
xx=np.linspace(-1,1,2*ngcols)



def clockXblocks1(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows1, nbcol, nbmux), order='F'),[1,0,2]), (nrows1 * nbcol, nbmux), order='F')

def blocksXtif1(data):
    """Translates `ccd` format to `row` format."""
    #return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2,:],2),data[:nrows1,:]),axis=1)
    return np.concatenate((np.rot90(data[nrows1+1+gap*2:2*(nrows1)+gap*2-1,:],2),data[2:nrows1,:]),axis=1)

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
def combine(data0, data1, thres=3e3):
    msk=data0<thres
    return (t12+1)*(data0*msk+data1)/(t12*msk+1)

######################3
# denoise bblocks
bpts=60
gg=np.exp(-(np.arange(-bpts//2,bpts//2)/(bpts/4))**2)
gg/=np.sum(gg)
#gg=np.reshape(gg,(bpts,1))

def conv2d(data,filt):
    data_s=np.empty(np.shape(data))
    nr=data.shape[1]

    for r in range(nr):
        data_s[:,r] = np.convolve(yy[:,r], gg, 'same')
    return data_s

def filter_bblocks(data):
    #yy=np.reshape(data[:,:,0],(485,192))
    yy=np.reshape(data[:,:,11],(485,192))
    bkgthr=5
    yy_s=conv2d(np.clip(yy,-bkgthr,bkgthr),gg)
    yy_s=np.reshape(yy_s,(485,192,1))
    #return data-yy_s
    #yavg = np.average(np.clip(bblocksXtif1(bkg[48,:,:])[0:10,:,:],-bkgthr,bkgthr),axis=0)
    data_out = data-yy_s
    data_out = data#-yy_s
    ###yy_avg=np.reshape(np.average(np.clip(bblocksXtif1(data_out)[1:11,:,:],0,2*bkgthr),axis=0),(1,192,12))
    yy_avg=np.reshape(np.average(np.clip(data_out[1:10,:,:],0,2*bkgthr),axis=0),(1,192,12))
    data_out -= yy_avg
    data_out *= data_out>2
    return data_out#-yy_s-yy_avg

######################3

def imgXraw_nofilter(data): # combine operations
    return imgXtif1(tif1Xbblocks(bblocksXtif1(data)))
  #  return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))


def imgXraw(data): # combine operations
    return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))

# Center of mass (COM):
xdim=nbpcol*nbmux//2
xx=np.reshape(np.arange(xdim),(xdim,1))
xx1=xx-(xx+1>xdim//2)*xdim

def center_of_mass(img):
    return np.array([np.sum(img*xx)/np.sum(img0),  np.sum(img0*xx.T)/np.sum(img0)])


smooth_factor=3
filter_width=frame_pixels//2
xx1c = np.fft.ifftshift(xx1)
rr1c = np.sqrt(np.fft.ifftshift(xx1**2+(xx1.T)**2))
sth=-np.fft.fftshift(np.arctan((rr1c-filter_width)/smooth_factor))/np.pi*2
sth*=sth*(sth>0)

#sth=np.fft.fftshift(np.arctan((xx1c+filter_width)/smooth_factor)/np.pi*2*np.arctan((-xx1c+filter_width)/smooth_factor)/np.pi*2)
#sth=np.fft.fftshift(np.arctan((xx1c+filter_width)/smooth_factor)/np.pi*2

#sth*=(sth>0)


def shift_img(img,shift,flt=sth):
    Fimg=np.fft.fft2(img)
    Fimg*=np.exp(1j*2*np.pi*shift[0]*xx1/xdim)*flt
    Fimg*=np.exp(1j*2*np.pi*shift[1]*xx1.T/xdim)*flt.T
    return np.real(np.fft.ifft2(Fimg))



from scipy import interpolate
coord=np.arange(-frame_pixels//2,frame_pixels//2)/frame_pixels*img_width+xdim//2
def rescale(img):
    return interpolate.interp2d(xx, xx, img, fill_value=0)(coord,coord)



import h5py


#from fccdv0 import blocksXtif, blocksXtif1, clockXblocks1

#---------------------------------
h5fname='raw_NS_200220033_002.cxi'

fid = h5py.File(h5fname, 'r')

# characterization of the dark field 

# get the background
bkg=np.array(fid['entry_1/data_1/dark_data'])

# split the average from 2 exposures:
bkg_avg0=np.average(bkg[0::2],axis=0)
bkg_avg1=np.average(bkg[1::2],axis=0)

rdata = fid['entry_1/data_1/raw_data']
n_frames = rdata.shape[0]//2 # number of frames (double-exposure)

# split short and long exposures
ii=2500//2+25
rdata0=rdata[ii*2]-bkg_avg0
rdata1=rdata[ii*2+1]-bkg_avg1

img0=combine(imgXraw(rdata0),imgXraw(rdata1))

# we need a shift, we take it from the first frame:
com = center_of_mass(img0*(img0>0))-xdim//2
#com = center_of_mass(img0*(img0>0))-xdim//2

def center_img(img):
    return shift_img(img0, com)



import matplotlib.pyplot as plt
img2 = center_img(img0)
plt.imshow(img2)
plt.figure(2)
img3 = rescale(img2)
plt.imshow(img3)
plt.draw()



# this is a copy of the raw data
# cp raw_NS_200220033_002.cxi filtered_NS_200220033.cxi 

h5fname_out='filtered_NS_200220033.cxi'
fido = h5py.File(h5fname_out, 'a')
try:
    del fido['entry_1/data_1/dark_data']
except:
    None
try:
    del fido['entry_1/data_1/raw_data']
except:
    None
    
try:
    del fido['entry_1/data_1/data']
except:
    None

out_data=fido.create_dataset("entry_1/data_1/data", (n_frames, frame_pixels,frame_pixels), dtype='f')


    # modify pixel size
del fido['entry_1/instrument_1/detector_1/x_pixel_size']
del fido['entry_1/instrument_1/detector_1/y_pixel_size']
fido['entry_1/instrument_1/detector_1/x_pixel_size']=x_pixel_size
fido['entry_1/instrument_1/detector_1/y_pixel_size']=x_pixel_size



import sys

Figon = False
#Figon = True

figure = None
for ii in np.arange(n_frames):
    #rdata0=rdata[0]
    img0 = combine(imgXraw(rdata[ii*2]-bkg_avg0),imgXraw(rdata[ii*2+1]-bkg_avg1))
    #img0 = combine(imgXraw_nofilter(rdata[ii*2]-bkg_avg0),imgXraw_nofilter(rdata[ii*2+1]-bkg_avg1))
    #img2 = shift_img(img0, com , flt=sth)
    img2 = center_img(img0)
    
    img3 = rescale(img2)
    out_data[ii] = img3
    #print('hello')
    sys.stdout.write('\r frame = %s/%s ' %(ii+1,n_frames))
    sys.stdout.flush()

    if Figon:
        if figure is None:
            figure = plt.imshow(np.abs(img3)**.2)
            ttl=plt.title
        else:
            figure.set_data(np.abs(img3)**.2)
        ttl('frame'+str(ii))
                
        plt.pause(.01)
        plt.draw()


fido.close()
fido = h5py.File(h5fname_out, 'r')

frames = fido['entry_1/data_1/data']
fido.close()


    
#def smooth(y):
#    y_smooth = np.convolve(y, gg, mode='same')
#    return y_smooth
#yy=(blocksXtif1(rdata1)[1:,12*50])
#yy_s=smooth(np.clip(yy,-5, 5))
##yy_s=smooth(yy)
#yy_s=np.reshape(yy_s,(485,1))
# 
#jk=np.reshape(yy[:,:,0],(485,192,1))

#yy=np.reshape((bblocksXtif1(rdata1))[:,:,0],(485,192,1))

yy=np.reshape((bblocksXtif1(rdata1))[:,:,0],(485,192))
bkgthr=5
yy_s=conv2d(np.clip(yy,-bkgthr,bkgthr),gg)
yy_s=np.reshape(yy_s,(485,192,1))
bblocksXtif1(rdata1)-yy_s
