#!/usr/bin/env python

import sys
import os
from cosmicp.options import parse_arguments
from cosmicp.common import rank, size, mpi_enabled, printd, printv, set_visible_device, complete_metadata, color, bcolors

if __name__ == '__main__':
    args = sys.argv[1:]

    options = parse_arguments(args)

    if options["gpu_accelerated"]:
        device_order, visible_devices, n_gpus = set_visible_device(rank)

        printd(color("Number of GPUs on this host = " + str(n_gpus), bcolors.HEADER))
        printd(color("GPU devices occupancy order: " + str(device_order), bcolors.HEADER))
        printd(color("GPU visible devices for this process = " + str(visible_devices), bcolors.HEADER))

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        printv(color("\r Running on CPU, enable -g option for a GPU execution", bcolors.HEADER))


    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" #This prevents JAX from taking over the whole device memory

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #this removes some log messages from tensorflow that can be pretty annoying

    from absl import logging
    logging.set_verbosity(logging.ERROR) #This does removes some annoying warnings from JAX


    import numpy as onp
    import jax.numpy as np
    import jax
    import cosmicp.diskIO as diskIO
    import cosmicp.preprocessor as preprocessor

    from cosmicp.diskIO import frames_out, map_tiffs, read_hdf5_group, read_metadata_hdf5
    from cosmicp.preprocessor import prepare, process, save_results
    from timeit import default_timer as timer

    print(options["fname"])

    #if options["fname"].endswith('.json'):

    metadata = diskIO.read_metadata(options["fname"])
    metadata = complete_metadata(metadata, options["conf_file"])
    dark_frames = diskIO.read_dark_data(metadata, options["fname"])
    ##########   
    base_folder = os.path.split(options["fname"])[:-1][0] + "/" 
    base_folder += os.path.basename(os.path.normpath(metadata["exp_dir"]))
    ##########
    raw_frames_tiff = map_tiffs(base_folder)

    #elif options["fname"].endswith('.h5')
    fname = ""

    metadata = read_metadata_hdf5(fname)
    metadata = complete_metadata(metadata, options["conf_file"])
    dark_frames_h = read_hdf5_group(fname, group = "entry_1/data_1/dark_frames")
    exp_frames_h = read_hdf5_group(fname, group = "entry_1/data_1/exp_frames")

    for i in range(0, dark_frames.shape[0]):

        print(dark_frames_h[:30])
        print(dark_frames[i] - dark_frames_h[i])

    for i in range(0, raw_frames_tiff.shape[0]):

        print(exp_frames_h[:30])
        print(raw_frames_tiff[i] - exp_frames_h[i])


    metadata, background_avg = prepare(metadata, dark_frames, raw_frames_tiff)
    out_data, my_indexes = process(metadata, raw_frames_tiff, background_avg, options["batch_size_per_rank"])

    save_results(options["fname"], metadata, out_data, my_indexes, metadata["translations"].shape[0])
  

