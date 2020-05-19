import numpy as np
import xcale
import xcale.common as ptycommon
from xcale.common.cupy_common import memcopy_to_device, memcopy_to_host
from xcale.common.misc import printd, printv
from xcale.common.cupy_common import check_cupy_available
from xcale.common.communicator import mpi_allGather, rank, size
import sys
import os
from PIL import Image
import diskIO

import baseline_filter
import stefano_filter

def calculateDecomposition(n_frames, my_rank, n_ranks):

    frames_per_rank = n_frames//n_ranks

    extra_work = 0

    if  my_rank == size + 1:
        extra_work =  data_shape[0] % n_ranks      

    printv("Frames to compute per rank: " + str(frames_per_rank))

    frames_range = range(my_rank * frames_per_rank, ((my_rank + 1) * frames_per_rank) + extra_work)

    printv("My range of ranks: " + str(frames_range))

    return frames_range


def preprocess(metadata, dark_frames, raw_frames, options, gpu_accelerated):

    input_data = {"raw_data": raw_frames, 
                  "dark_data": dark_frames}

    if gpu_accelerated:

        if options["debug"]: printv("Using GPU acceleration: allocating and transfering pointers from host to GPU...")
    
        xp = __import__("cupy") 
        mode = "cuda"

        memcopy_to_device(input_data)     

    else:
        xp = __import__("numpy")
        mode = "python"


    filter_local_data = baseline_filter.allocate(input_data["raw_data"].shape, xp)
    filter_local_data = baseline_filter.initialize(filter_local_data, xp)
    output_data = baseline_filter.filter(filter_local_data, input_data, metadata, mode, xp) #This will generate a set of clean frames

    if gpu_accelerated:

        if options["debug"]: printv("Using GPU acceleration: transfering pointers back from GPU to host...")
        memcopy_to_host(output_data)  

    return output_data

#We run this guy as:
#python preprocess.py raw_NS_200220033_002.cxi

if __name__ == '__main__':

    args = sys.argv[1:]

    options = {"debug": True,
               "gpu_accelerated": False}

    gpu_available = check_cupy_available()

    if not ptycommon.communicator.mpi_enabled:
        printd("\nWARNING: mpi4py is not installed. MPI communication and partition won't be performed.\n" + \
               "Verify mpi4py is properly installed if you want to enable MPI communication.\n")

    if not gpu_available and options["gpu_accelerated"]:
        printd("\nWARNING: GPU mode was selected, but CuPy is not installed.\n" + \
               "Verify CuPy is properly installed to enable GPU acceleration.\n" + \
               "Switching to CPU mode...\n")

        options["gpu_accelerated"] = False


    json_file = args[0]

    
    metadata, dark_frames, raw_frames = diskIO.read(json_file)

    metadata["final_res"] = 5e-9 # desired final pixel size nanometers

    metadata["desired_padded_input_frame_width"] = None

    metadata["output_frame_width"] = 256 # final frame width 

    n_frames = metadata["translations"].shape[0]

    # ii is the middle frame, we could use the first or average a few TODO: why + sqrt()? that is not the middle, is it
    center = np.int(n_frames + np.sqrt(n_frames))//2

    if metadata["double_exposure"]:

        metadata["double_exp_time_ratio"] = metadata["dwell1"] / metadata["dwell2"] # time ratio between long and short exposure

        center_frames = np.array([raw_frames[ii*2], raw_frames[ii*2 + 1]])

    else:

        center_frames = raw_frames[ii*2]


    metadata["detector_pixel_size"] = 30e-6  # 30 microns
    metadata["detector_distance"] = 0.0121 #this is in meters

    my_indexes = np.array(calculateDecomposition(metadata["translations"].shape[0], rank, size))


    print(metadata)
    print(dark_frames)
    print(raw_frames)

    metadata, background_avg =  stefano_filter.prepare(metadata, center_frames, dark_frames):
    output_data = stefano_filter.process_stack(metadata, raw_frames, background_avg):

    #output_data = preprocess(metadata, dark_frames, raw_frames, options, options["gpu_accelerated"])

    data_dictionary = {}
    data_dictionary.update({"data" : output_data["preproc_data"]})

    output_filename = os.path.splitext(json_file)[:-1][0] + "_preproc.cxi"

    print("Saving cxi file: " + output_filename)

    io = ptycommon.IO()


    if rank is 0: #there is still no merge of the data, this guy only would write his partial results

        #This script deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
        io.write(output_filename, data_dictionary, data_format = io.dataFormat) #We use the metadata we readed above and drop it into the new cxi

