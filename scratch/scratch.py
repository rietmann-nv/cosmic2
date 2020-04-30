#!/usr/bin/env python3
# -*- coding: utf-8 -*-

E = 1300 #eV
hnu = E* 1.15e-2 #
ccd_pixel=30e-6

frame_pixels=256 # final pixels 
img_width= 600 # cropped width of the raw clean frames

# fccd readout
nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

# shifting/cropping frames
nrcols=487
ngcols=480
nrows = 520
gap = 33
nrows1=nrows-gap # good rows



# pixel size is rescaled
x_pixel_size=ccd_pixel*img_width/frame_pixels



import numpy as np
xx=np.linspace(-1,1,2*ngcols)


def clockXblocks1(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows1, nbcol, nbmux), order='F'),[1,0,2]), (nrows1 * nbcol, nbmux), order='F')

#plt.imshow(imgXtif1(tif1Xbblocks(bblocksXtif1(rdata1))))
def blocksXtif1(data):
    """Translates `ccd` format to `row` format."""
    return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2,:],2),data[:nrows1,:]),axis=1)

def bblocksXtif1(data):
    return np.reshape(blocksXtif1(data),(nrcols,nbmux,nbcol))

def tif1Xbblocks(data):
    return np.reshape(data[:,:,1:nbcol-1],(nrcols,nbmux*nbpcol))

def imgXtif1(data):
    return np.concatenate((data[6:486,0:960],np.rot90(data[6:486,960:],2)))
#def imgX

def imgXraw(data):
    return imgXtif1(tif1Xbblocks(bblocksXtif1(data)))
        

def combine(data0, data1, t12 = 5, thres=3e3):
    msk=data0<thres
    return (data0*msk+data1)/(t12*msk+1)

# Center of mass (COM):
xdim=nbpcol*nbmux//2
xx=np.reshape(np.arange(xdim),(xdim,1))
xx1=xx-(xx+1>xdim//2)*xdim

def center_of_mass(img):
    return np.array([np.sum(img*xx)/np.sum(img0),  np.sum(img0*xx.T)/np.sum(img0)])


smooth_factor=3
filter_width=128
xx1c = np.fft.ifftshift(xx1)
sth=np.fft.fftshift(np.arctan((xx1c+filter_width)/smooth_factor)/np.pi*2*np.arctan((-xx1c+filter_width)/smooth_factor)/np.pi*2)
sth*=(sth>0)

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

# get the background
bkg=np.array(fid['entry_1/data_1/dark_data'])

# split the average from 2 exposures:
bkg_avg0=np.average(bkg[0::2],axis=0)
bkg_avg1=np.average(bkg[1::2],axis=0)

rdata = fid['entry_1/data_1/raw_data']

# split short and long exposures
rdata0=rdata[0]-bkg_avg0
rdata1=rdata[1]-bkg_avg1

img0=combine(imgXraw(rdata0),imgXraw(rdata1))

# we need a shift, we take it from the first frame:
com = center_of_mass(img0*(img0>0))-xdim//2

img2 = np.real(shift_img(img0, com))
img3 = rescale(img2)

plt.imshow(img2)
plt.figure(2)
plt.imshow(img3)
plt.draw()



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

    # modify pixel size
del fido['entry_1/instrument_1/detector_1/x_pixel_size']
del fido['entry_1/instrument_1/detector_1/y_pixel_size']
fido['entry_1/instrument_1/detector_1/x_pixel_size']=x_pixel_size
fido['entry_1/instrument_1/detector_1/y_pixel_size']=x_pixel_size


n_frames = rdata.shape[0]//2

out_data=fido.create_dataset("entry_1/data_1/data", (n_frames, frame_pixels,frame_pixels), dtype='f')

img_stack=np.empty((10,frame_pixels,frame_pixels))

#img = None
import sys
def printbar(percent,string='  '):        
        #percent = ((position + 1) * 100) // (n_tasks + n_workers)
        #sys.stdout.write(
        #    '\rProgress: [%-50s] %3i%% ' %
        #    ('=' * (percent // 2), percent))
        
        sys.stdout.write('\r%s Progress: [%-50s] %3i%% ' %(string,'=' * (percent // 2), percent))
        sys.stdout.flush()

for ii in np.arange(n_frames):
    #rdata0=rdata[0]
    img0 = combine(imgXraw(rdata[ii*2]-bkg_avg0),imgXraw(rdata[ii*2+1]-bkg_avg1))
    img2 = np.real(shift_img(img0, com-xdim//2,flt=sth))
    img3 = rescale(img2)
    out_data[ii] = img3
    #print('hello')
    sys.stdout.write('\r frame = %s/%s ' %(ii+1,n_frames))
    sys.stdout.flush()

    if False:
        figure = None
        if figure is None:
            figure = plt.imshow(img3)
            ttl=plt.title
        else:
            figure.set_data(img3)
        ttl('frame'+str(ii))
                
        plt.pause(.01)
        plt.draw()

fido.close()
fido = h5py.File(h5fname_out, 'r')

frames = fido['entry_1/data_1/data']
