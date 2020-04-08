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

import baseline_filter

def calculateDecomposition(n_frames, my_rank, n_ranks):

    frames_per_rank = n_frames//n_ranks

    extra_work = 0

    if  my_rank == size + 1:
        extra_work =  data_shape[0] % n_ranks      

    printv("Frames to compute per rank: " + str(frames_per_rank))

    frames_range = range(my_rank * frames_per_rank, ((my_rank + 1) * frames_per_rank) + extra_work)

    printv("My range of ranks: " + str(frames_range))

    return frames_range


def preprocess(raw_data, dark_data, metadata, options, gpu_accelerated):

    input_data = {"raw_data": raw_data, 
                  "dark_data": dark_data}

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


    cxi_name = args[0]

    io = ptycommon.IO()
    metadata = io.read(cxi_name, io.metadataFormat) #This guy has all information needed

    n_exposures = metadata["n_exposures"]

    my_indexes = np.array(calculateDecomposition(metadata["translations"].shape[0], rank, size))

    data = {}

    data = {**data, **io.read(cxi_name, {"raw_data": ptycommon.IO().dataFormat["raw_data"]}, my_indexes)}
    data = {**data, **io.read(cxi_name, {"dark_data": ptycommon.IO().dataFormat["dark_data"]})} #Each rank reads all darks for now

    output_data = preprocess(data["raw_data"], data["dark_data"], metadata, options, options["gpu_accelerated"])

    data_dictionary = {}
    data_dictionary.update({"data" : output_data["preproc_data"]})

    output_filename = "preproc_" + os.path.basename(cxi_name)

    print("Saving cxi file: " + output_filename)

    if rank is 0: #there is still no merge of the data, this guy only would write his partial results

        #This script deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
        io.write(output_filename, data_dictionary, data_format = io.dataFormat) #We use the metadata we readed above and drop it into the new cxi

