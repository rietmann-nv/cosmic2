# hello

import os
#You can also go with os.environ["JAX_PLATFORM_NAME"] = 'cpu', but for some reason that still reserves some GPU memory even though it apparently runs only on CPU. 
#With this way GPUs are not touched
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax.numpy as jnp
import jax

import cupy

import cosmicp.fccd as fccd ##this module filters the data
import cosmicp.preprocessor as preprocessor ##this module mostly does data handling
import cosmicp.fccd as fccd_jax
import cosmicp.preprocessor as preprocessor_jax

import cosmicp.diskIO as diskIO ##reads TIFs from disk in chunks
from skimage.io import imread

import matplotlib.pyplot as plt
from timeit import default_timer as timer

import numpy as np

import IPython



def numpy_pipeline(darkFrames, longTIF, shortTIF, metadata):

    ##average the frames for long and short exposures and output each
    ##the dark stripe in the middle are non-physical pixels.  There are additional non-physical pixels in the
    ##two halves which must be removed
    bkgAve = preprocessor.split_background(darkFrames)
    longFrame = longTIF - bkgAve[0]
    shortFrame = shortTIF - bkgAve[1]
    longFrameStripped = fccd.bblocksXtif1(longFrame)
    shortFrameStripped = fccd.bblocksXtif1(shortFrame)
    ##this function smooths the pixels below a threshold thus flattening out the background readout noise
    longFrameFiltered = fccd.filter_bblocks(longFrameStripped)
    shortFrameFiltered = fccd.filter_bblocks(shortFrameStripped)

    ##this function just reorders the data to properly interleave the ADC channels
    longFrameFiltered = fccd.tif1Xbblocks(longFrameFiltered)
    shortFrameFiltered = fccd.tif1Xbblocks(shortFrameFiltered)

    ##final reordering of the frame to match the actual CCD geometry
    ##long frame is saturated in the center
    longFrameReordered = fccd.imgXtif1(longFrameFiltered)
    shortFrameReordered = fccd.imgXtif1(shortFrameFiltered)
    finalFrame = preprocessor.combine_double_exposure(longFrameReordered, shortFrameReordered, metadata["dwell1"] // metadata["dwell2"])

    return finalFrame


def jax_frame_pipeline(TIF, bkgAve):
    frame = TIF - bkgAve
    frameStripped = fccd_jax.bblocksXtif1(frame)
    ##this function smooths the pixels below a threshold thus flattening out the background readout noise
    frameFiltered = fccd_jax.filter_bblocks(frameStripped)

    ##this function just reorders the data to properly interleave the ADC channels
    frameFiltered = fccd_jax.tif1Xbblocks(frameFiltered)

    ##final reordering of the frame to match the actual CCD geometry
    ##long frame is saturated in the center
    return fccd_jax.imgXtif1(frameFiltered)

def jax_pipeline(darkFrames, longTIFs, shortTIFs, metadata):
    """
    Version of the code that uses JAX's batching operator `vmap`, which often leads to much higher performance.
    """
    bkgAve = preprocessor_jax.split_background(darkFrames)
    frame_pipeline_vmap0 = jax.vmap(lambda TIF: jax_frame_pipeline(TIF, bkgAve[0]))
    frame_pipeline_vmap1 = jax.vmap(lambda TIF: jax_frame_pipeline(TIF, bkgAve[1]))
    
    longTIFs_processed = frame_pipeline_vmap0(longTIFs)
    shortTIFs_processed = frame_pipeline_vmap1(shortTIFs)

    f_cde_vmap = jax.vmap(lambda lTp, sTp: preprocessor_jax.combine_double_exposure(lTp, sTp, metadata["dwell1"] // metadata["dwell2"]))

    return f_cde_vmap(longTIFs_processed, shortTIFs_processed)

def main():


    print("Loading data")
    ##metadata is saved by the control system into a JSON file.  Analysis code reads from here

    jsonFile = '/home/pablo/ALS_preprocessor_data/190602068/190602068_002_info.json'
    #jsonFile = '../Data/NS_200805056/200805056/200805056_002_info.json'
    metadata = diskIO.read_metadata(jsonFile)
    ##load the dark frames from disk.  These need to be averaged for processing of data frames.
    darkFrames = diskIO.read_dark_data(metadata,jsonFile)

    ##once the background frames are averaged, several processes will grab the many data frames in chunks
    ##for this demonstration I'll just grab a single frame (actually, two: long and short exposure) to
    ##show the processing.  Frames are named image000000.tif, image000001.tif, etc.  Even numbers are long
    ##exposure and odd numbers are short exposure
    ##background TIFs are in directory 001 and data TIFs in 002
    longTIFile = '/home/pablo/ALS_preprocessor_data/190602068/002/image000000.tif'
    shortTIFile = '/home/pablo/ALS_preprocessor_data/190602068/002/image000001.tif'

    #longTIFile = '../Data/NS_200805056/200805056/002/image000000.tif'
    #shortTIFile = '../Data/NS_200805056/200805056/002/image000001.tif'


    longTIF = imread(longTIFile)
    shortTIF = imread(shortTIFile)

    # get shape of final image:
    final_img_shape = numpy_pipeline(darkFrames, longTIF, shortTIF, metadata).shape

    NUM_IMAGES = 10

    # synthetic way to have more images
    longTIFs = np.repeat(longTIF[np.newaxis, :, :], NUM_IMAGES, axis=0)
    shortTIFs = np.repeat(shortTIF[np.newaxis, :, :], NUM_IMAGES, axis=0)
    all_final_img_shape = (longTIFs.shape[0], final_img_shape[0], final_img_shape[1])
    finalFrames = np.zeros(all_final_img_shape)
    print("Running NumPy version")

    start = timer()
    for i in range(longTIFs.shape[0]):
        finalFrames[i, :, :] = numpy_pipeline(darkFrames, longTIFs[i, :, :], shortTIFs[i, :, :], metadata)
    
    end = timer()
    # result on 1080Ti: NumPy (100 images): total 10.29498144495301s, 0.1029498144495301s / img
    print("NumPy ({} images): total {}s, {}s / img".format(NUM_IMAGES, end-start, (end-start)/NUM_IMAGES))

    # transfer to GPU
    start = timer()
    d_darkFrames = jnp.array(darkFrames)
    end = timer()
    print("CPU->GPU Xfer darkframes: ", end-start)
    start = timer()
    d_longTIFs = jnp.array(longTIFs)
    d_shortTIFs = jnp.array(shortTIFs)
    end = timer()
    print("CPU->GPU Xfer long/short TIF pair ({} images): {}s, {}s / img pair".format(NUM_IMAGES, end-start, (end-start)/NUM_IMAGES))
    
    
    fvmap = lambda darkFrames, longTIF, shortTIF: jax_pipeline(darkFrames, longTIF, shortTIF, metadata)
    fjit_vmap = jax.jit(fvmap)

    # Jax traces the expressions and compiles to GPU at runtime, so we use this step as a "warm up" for our benchmarks
    finalFrames_jax_vmap = fjit_vmap(d_darkFrames, d_longTIFs, d_shortTIFs)

    # JAX timing:
    start = timer()
    cupy.cuda.nvtx.RangePush("fjit second time: vmap")
    finalFrames_jax_vmap = fjit_vmap(d_darkFrames, d_longTIFs, d_shortTIFs).block_until_ready()
    cupy.cuda.nvtx.RangePop()
    end = timer()
    # Results on 1080Ti: Jax vmap (100 images): 0.0467481940286234s, 0.000467481940286234s / img
    print("Jax vmap ({} images): {}s, {}s / img".format(NUM_IMAGES, end-start, (end-start)/NUM_IMAGES))

    _, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.matshow(np.log(finalFrames[0, : ,:] + 1.))
    ax1.set_title("NumPy")
    ax2.matshow(np.log(finalFrames_jax_vmap[0, :, :] + 1.))
    ax2.set_title("JAX (vmap)")
    ax3.matshow(np.abs(finalFrames[0, :, :] - finalFrames_jax_vmap[0, :, :]))
    ax3.set_title("diff")
    plt.show()
    
    np.testing.assert_allclose(finalFrames, finalFrames, rtol=5e-4, atol=10)
    np.testing.assert_allclose(finalFrames, finalFrames_jax_vmap, rtol=5e-4, atol=10)
    
if __name__ == "__main__":
    main()
