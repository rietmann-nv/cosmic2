#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py
import numpy as np

#from fccdv0 import blocksXtif, blocksXtif1, clockXblocks1

h5fname='raw_NS_200220033_002.cxi'

fid = h5py.File(h5fname, 'r')

#a=fid['entry_1/data_1']

#bkg=fid['entry_1/data_1/raw_data']
bkg=fid['entry_1/data_1/dark_data']

#fi

bkg=np.array(bkg)

bkg.shape

bkg_avg=np.average(bkg,axis=0)

bkg_avg1=np.average(bkg[1::2],axis=0)
bkg_avg0=np.average(bkg[0::2],axis=0)


rdata = fid['entry_1/data_1/raw_data']

#rdata2=rdata[0]-bkg_avg0

#rdata1=rdata[1]-bkg_avg0

nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

nrcols=487
ngcols=480
nrows = 520
gap = 33
nrows1=nrows-gap # good rows


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
    

    
rdata1=rdata[1]-bkg_avg1
rdata0=rdata[0]-bkg_avg0

def combine(data0, data1, t12 = 5, thres=3e3):
    msk=data0<thres
    return (data0*msk+data1)/(t12*msk+1)
    
plt.imshow(combine(imgXraw(rdata0),imgXraw(rdata1),thres=3.5e3))

#data=blocksXtif1(bkg_avg)
#data1=clockXblocks1(data)

#rdata=fid['entry_1/data_1/raw_data']

#tname='/tomodata/NS/200220033/002/image000448.tif'

