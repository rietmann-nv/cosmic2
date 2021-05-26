#!/usr/bin/env python

import os
#You can also go with os.environ["JAX_PLATFORM_NAME"] = 'cpu', but for some reason that still reserves some GPU memory even though it apparently runs only on CPU. 
#With this way GPUs are not touched
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as onp
import jax.numpy as np
import jax
from cosmicp.common import rank, size, mpi_enabled, printd, printv, set_visible_device
from cosmicp.options import parse_arguments
import sys
import os
import cosmicp.diskIO as diskIO
import cosmicp.preprocessor as preprocessor

import IPython

from cosmicp.diskIO import frames_out, map_tiffs
from timeit import default_timer as timer

def convert_translations(translations):

    zeros = np.zeros(translations.shape[0])

    new_translations = np.column_stack((translations[:,1]*1e-6,translations[::-1,0]*1e-6, zeros)) 

    return new_translations


def preprocessing_pipeline(metadata, dark_frames, raw_frames):

    raw_frames = np.array(raw_frames)
    dark_frames = np.array(dark_frames)

    n_frames = raw_frames.shape[0]
    n_total_frames = metadata["translations"].shape[0]

    #This takes the center of the stack as a center frame(s)
    center = int(n_total_frames)//2

    #Check this in case we are in double exposure
    if center % 2 == 1:
        center -= 1

    if metadata["double_exposure"]:
        metadata["double_exp_time_ratio"] = metadata["dwell1"] // metadata["dwell2"] # time ratio between long and short exposure
        center_frames = np.array([raw_frames[center], raw_frames[center + 1]])
    else:
        center_frames = raw_frames[center]
    
    
    #print('energy (eV)',metadata['energy'])
    
    metadata, background_avg =  preprocessor.prepare(metadata, center_frames, dark_frames)


    data_shape = (n_frames//(metadata['double_exposure']+1), metadata["output_frame_width"], metadata["output_frame_width"])
    out_frames = np.zeros(data_shape)

    output_data = preprocessor.process_stack(metadata, raw_frames,background_avg, out_frames)

    return output_data



if __name__ == '__main__':
    args = sys.argv[1:]

    options = parse_arguments(args)

    if not mpi_enabled:
        printd("\nWARNING: mpi4py is not installed. MPI communication and partition won't be performed.\n" + \
               "Verify mpi4py is properly installed if you want to enable MPI communication.\n")


    if options["gpu_accelerated"]:
        device_order, visible_devices, n_gpus = set_visible_device(rank)

        printd("Number of GPUs on this host = " + str(n_gpus))
        printd("GPU devices occupancy order: " + str(device_order))
        printd("GPU visible devices for this process = " + str(visible_devices))

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        printd("Running on CPU, enable -g option to run on GPU")


    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #this removes some log messages from jax/tensorflow but apparently it is not working
    import jax
    import cosmicp.preprocessor as preprocessor

    metadata = diskIO.read_metadata(options["fname"])

    #These do not change
    metadata["detector_pixel_size"] = 30e-6  # 30 microns
    metadata["detector_distance"] = 0.121 #this is in meters

    #These here below are options, we can edit them
    metadata["final_res"] = 3e-9 # desired final pixel size nanometers
    metadata["desired_padded_input_frame_width"] = None
    metadata["output_frame_width"] = 256 # final frame width 
    metadata["translations"] = convert_translations(metadata["translations"])
    #Energy from eV to Joules
    #import scipy.constants
    #metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge



    n_total_frames = metadata["translations"].shape[0]
    if metadata["double_exposure"]: n_total_frames *= 2

    dark_frames = diskIO.read_dark_data(metadata, options["fname"])

    ##########   

    base_folder = os.path.split(options["fname"])[:-1][0] + "/" 
    base_folder += os.path.basename(os.path.normpath(metadata["exp_dir"]))
  
    ##########
    raw_frames_tiff = map_tiffs(base_folder)
    num_frames = raw_frames_tiff.shape[0]
    framesize = raw_frames_tiff['0'].shape

    print("Number of frames[{}]: {}, framesize: {}, estimated GB: {}".format(base_folder, num_frames, framesize,
                                                                             num_frames * framesize[0] * framesize[1] * 8 / 1e9))
    
    

    raw_frames = np.zeros((num_frames, framesize[0], framesize[1]))
    for i in range(num_frames):
        raw_frames = jax.ops.index_update(raw_frames, jax.ops.index[i, :, :], raw_frames_tiff[str(i)])

    # ....
    start = timer()
    output_data, _ = preprocessing_pipeline(metadata, dark_frames, raw_frames)
    end = timer()
    print("Preprocessing time: ", end-start)

    print("output_data size: {}".format(output_data.shape))

    io = diskIO.IO()
    output_filename = os.path.splitext(options["fname"])[:-1][0][:-4] + "cosmic2.cxi"

    if rank == 0:

        #output_filename = os.path.splitext(options["fname"])[:-1][0] + "_cosmic2.cxi"

        printv("\nSaving cxi file metadata: " + output_filename + "\n")

        #data_dictionary["data"] = np.concatenate(data_dictionary["data"], axis=0)

        #This script deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        #import time
        #time.sleep(10)

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
        #io.write(output_filename, data_dictionary, data_format = io.dataFormat) #We use the metadata we read above and drop it into the new cxi


    data_shape = output_data.shape

    if rank == 0:
        out_frames, fid = frames_out(output_filename, data_shape)  
    else:
        out_frames = 0
    

    # write to HDF5 file
    out_frames[:, :, :] = output_data[:, :, :]


    del metadata["hdr_path"]
    del metadata["dark_dir"]
    del metadata["exp_dir"]

    
    if rank ==0:
        fid.close()
  

