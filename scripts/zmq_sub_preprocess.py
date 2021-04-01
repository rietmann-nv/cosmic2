

import zmq
import numpy as np
import cupy as cp
import jax.numpy as jnp
import jax
import tifffile
import os
from cosmicp.fccd_alt import FCCD
from cosmicp.fccd_alt_jax import FCCD as FCCD_jax

import matplotlib.pyplot as plt

import time

addr = 'tcp://127.0.0.1:49206'
shape = (1040, 1152)
path = './tmp/images/'

# don't let JAX use all the GPU memory!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

num_rows = shape[0] // 2
num_adcs = shape[1] // 6
CCD = FCCD(nrows=num_rows)
CCD_jax = FCCD_jax(nrows=num_rows)

context = zmq.Context()
frame_socket = context.socket(zmq.SUB)
frame_socket.setsockopt(zmq.SUBSCRIBE, b'')
frame_socket.set_hwm(2000)
frame_socket.connect(addr)
if not os.path.exists(path):
    os.makedirs(path)

assemble2_jax = jax.jit(CCD_jax.assemble2)

while True:
    num, array_len_bytes, cp_dtype_bytes, handle = frame_socket.recv_multipart()  # blocking
    array_len = int(array_len_bytes)
    cp_dtype_str = cp_dtype_bytes.decode("ascii")

    ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)
    mem = cp.cuda.UnownedMemory(ptr, array_len, context)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    npbuf_gpu = cp.ndarray((array_len,), cp.dtype(cp_dtype_str), memptr)
    print("Reading file with len: ", array_len)
    # npbuf = cp.asnumpy(npbuf_gpu)

    npbuf_gpu = npbuf_gpu.reshape((12 * num_rows, num_adcs))
    image = CCD.assemble2(cp.asnumpy(npbuf_gpu.astype(cp.uint16)))
    npbuf_jax= jnp.array(cp.asnumpy(npbuf_gpu.astype(cp.uint16)))
    image_jax = assemble2_jax(npbuf_jax)

    print("Diff: ", np.sum(np.array(image_jax) - image))

    # plt.imshow(image)
    # plt.draw()
    # plt.pause(0.01)
    # print("Plotting", num)

    
    # imfile = path+'/image%06d.tif' % int(num)
    # print('Saving to ' + imfile)
    # tifffile.imsave(imfile, image)
