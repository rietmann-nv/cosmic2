#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:17:15 2019

@author: smarchesini
"""
 
#pylab
from matplotlib.pylab import plt

import scipy
import numpy as np

def animate(data, mwidth=1, mshrp=7):
    img=None
    #num_rays = data.shape[2]
    #flt=fltr(num_rays, w=mwidth, s=mshrp)
    #flt.shape=(1,num_rays)
    
    n_frames = data.shape[0]
    for ii in range(n_frames):
        im = data[ii,:,:]#*flt
        if img is None:
            img = plt.imshow(im)
            ttl=plt.title
        else:
            img.set_data(im)
        ttl('frame'+str(ii)+'/'+str(n_frames))
        plt.pause(.01)
        plt.draw()
        
def fltr(num_rays, w=.9, s=7):
    xx=abs(np.linspace(-1,1,num=num_rays))
    flt=((scipy.special.erf((w-xx)*s)+1)/2)**2
    return flt
