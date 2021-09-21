#!/usr/bin/env python

import sys
import os
import h5py
import zmq
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

    from cosmicp.diskIO import frames_out, map_tiffs, read_metadata_hdf5
    from cosmicp.preprocessor import prepare, process, save_results, receive_metadata, subscribe_to_socket
    from timeit import default_timer as timer

    options["input_address"] = options["fname"] 
    network_metadata = {}

    #data coming from socket
    if "input_address" in options and options["input_address"] != None:

        network_metadata["input_address"] = options["input_address"]

        network_metadata["context"] = zmq.Context()

        network_metadata["input_socket"] = subscribe_to_socket(network_metadata)

        metadata = receive_metadata(network_metadata)
        metadata = complete_metadata(metadata, options["conf_file"])

        dark_frames, exp_frames = [],[]
        print(metadata)

    elif options["fname"].endswith('.json'):
        #fname = "/cosmic-dtn/groups/cosmic/Data/2021/09/210916/210916006/210916006_002_info.json"
        metadata = diskIO.read_metadata(options["fname"])
        metadata = complete_metadata(metadata, options["conf_file"])

        print(metadata)

        dark_frames = diskIO.read_dark_data(metadata, options["fname"])
        ##########   
        base_folder = os.path.split(options["fname"])[:-1][0] + "/" 
        base_folder += os.path.basename(os.path.normpath(metadata["exp_dir"]))
        ##########
        exp_frames = map_tiffs(base_folder)

    elif options["fname"].endswith('.h5'):
        #fname = "/cosmic-dtn/groups/cosmic/Data/2021/09/210916/210916003/raw_data.h5"

        metadata = read_metadata_hdf5(options["fname"])
        metadata = complete_metadata(metadata, options["conf_file"])

        print(metadata)

        f = h5py.File(options["fname"], 'r')

        dark_frames = f["entry_1/data_1/dark_frames"]
        exp_frames = f["entry_1/data_1/exp_frames"]


    metadata, background_avg, received_exp_frames = prepare(metadata, dark_frames, exp_frames, network_metadata)
    out_data, my_indexes = process(metadata, exp_frames, background_avg, options["batch_size_per_rank"], received_exp_frames, network_metadata)

    f.close

    if "input_address" in options and options["input_address"] != None:
        network_metadata["input_socket"].disconnect(options["input_address"])

    save_results(options["fname"], metadata, out_data, my_indexes, metadata["translations"].shape[0])
  

