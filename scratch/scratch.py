#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py
import numpy as np

from fccdv0 import blocksXtif, blocksXtif1, clockXblocks1

h5fname='raw_NS_200220033_002.cxi'

fid = h5py.File(h5fname, 'r')

#a=fid['entry_1/data_1']

bkg=fid['entry_1/data_1/raw_data']
#fi

bkg=np.array(bkg)

bkg.shape

bkg_avg=np.average(bkg,axis=0)

bkg_avg1=np.average(bkg[1::2],axis=0)
bkg_avg0=np.average(bkg[0::2],axis=0)



data=blocksXtif1(bkg_avg)
data1=clockXblocks1(data)

rdata=fid['entry_1/data_1/raw_data']

#rdata2=rdata[0]-bkg_avg0

#rdata1=rdata[1]-bkg_avg0

nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

nrcols=487
ngcols=480



def bblocksXtif1(data):
    return np.reshape(blocksXtif1(data),(nrcols,nbmux,nbcol))

def tif1Xbblocks(data):
    return np.reshape(data[:,:,1:nbcol-1],(nrcols,nbmux*nbpcol))

def imgXtif1(data):
    return np.concatenate((data[6:486,0:960],np.rot90(data[6:486,960:],2)))
#def imgX


#rdatab=np.reshape(blocksXtif1(rdata1),(nrcols,nbmux,nbcol))
#rdatac=np.reshape(rdatab[:,:,1:nbcol-1],(nrcols,nbmux*nbpcol))

#rdatad=np.concatenate((rdatac[:,0:960],np.rot90(rdatac[:,960:],2)))
#import pylab

