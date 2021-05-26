
import sys
import subprocess
import operator
import os

try: 
    from mpi4py import *
    from mpi4py import MPI
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


def printd(string):
    if mpi_enabled:
        print("MPI Rank " + str(rank) + ": " + str(string))
    else: 
        printv(string)

def printv(string):
    if rank is 0:
        print(string)

def gather(local, shape, dtype):

    sendbuf = np.array(local)

    if rank == 0:
        #print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(shape, dtype)
        print(recvbuf.shape)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf, recvbuf=recvbuf)

    return recvbuf

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
    req=comm.Igatherv(sendbuf=[np.asarray(data_local), mpichunktype],recvbuf=[data,(cnt*sdim,None),mpichunktype])
    mpichunktype.Free()
    
    return req

def mpi_allGather(send_buff, recv_buff):

    global size, rank, mpi_enabled


    if size > 1 and mpi_enabled:
        recv_buff = MPI.COMM_WORLD.allgather(send_buff)
    else:
        recv_buff = [send_buff]

    return recv_buff

#Setup local GPU device based on a gpu_priority integer. The lower the priority, the more available gpu this assigns.
def set_visible_device(gpu_priority):

    cuda_dir = ""
    try:
        cmd = "which nvcc"
        cuda_dir = str(subprocess.check_output(cmd.split(" "))).replace("bin/nvcc\\n'", "").replace("b'", "")
    except: pass

    cuda_dir_list = []

    cuda_dir_list = mpi_allGather(cuda_dir, cuda_dir_list)

    #printd(', '.join(str(e) for e in cuda_dir_list))

    for i in cuda_dir_list:
        if i is not "":    
            cuda_dir = i
            break

    os.environ['CUDA_PATH'] = cuda_dir

    printd("CUDA directory: " + cuda_dir)

    cmd = "nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used,pstate,utilization.gpu --format=csv"
 
    result = str(subprocess.check_output(cmd.split(" ")))

    if sys.version_info <= (3, 0):
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split()[1:] # use this for python2
    else:
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split("\\n")[1:] # use this for python3


    result = [res.split(",") for res in result]
  
    if sys.version_info >= (3, 0):
        result = result[0:-1]

    result = [[int(r[0]), int(r[3])] for r in result]
    result = sorted(result, key=operator.itemgetter(1))

    n_devices = len(result)

    nvidia_device_order = [res[0] for res in result]

    visible_devices = str(nvidia_device_order[(gpu_priority % len(nvidia_device_order))])
      
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    return nvidia_device_order, visible_devices, n_devices
