
import sys

try: 
    from mpi4py import *
    from mpi4py import MPI
except ImportError: pass

try: 
    import cupy as cp
except ImportError: pass

import numpy as np

mpi_enabled = "mpi4py" in sys.modules

comm = None
if mpi_enabled:
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
else:
    size = 1
    rank = 0


def check_cupy_available():

    available = False

    try:
        import cupy as cp
        available = True
    except ImportError:
        pass
  
    return available


def printd(string):
    if mpi_enabled:
        print("MPI Rank " + str(rank) + ": " + str(string))
    else: 
        printv(string)

def printv(string):
    if rank is 0:
        print(string)


def igatherv(data_local,chunk_slices, data = None): 
    if size==1: 
        if type(data) == type(None):
            data=data_local+0
        else:
            data[...] = data_local[...]
        return data

    cnt=np.diff(chunk_slices)
    slice_shape=data_local.shape[1:]
    sdim=np.prod(slice_shape)
    
    if rank==0 and type(data) == type(None) :
        tshape=(np.append(chunk_slices[-1,-1]-chunk_slices[0,0],slice_shape))
        data = np.empty(tuple(tshape),dtype=data_local.dtype)


    #comm.Gatherv(sendbuf=[data_local, MPI.FLOAT],recvbuf=[data,(cnt*sdim,None),MPI.FLOAT],root=0)
    
    # for large messages
    mpichunktype = MPI.FLOAT.Create_contiguous(4).Commit()
    #mpics=mpichunktype.Get_size()
    sdim=sdim* MPI.FLOAT.Get_size()//mpichunktype.Get_size()
    req=comm.Igatherv(sendbuf=[data_local, mpichunktype],recvbuf=[data,(cnt*sdim,None),mpichunktype])
    mpichunktype.Free()
    
    return req