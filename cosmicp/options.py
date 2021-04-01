#!/usr/bin/env python

import sys, getopt
from cosmicp.common import printv


help =   "\n\cosmicp.py [options] input.json\n\n\
-g -> \tEnable GPU acceleration, off by default.\n\
\n\n"

def parse_arguments(args, options = None):

    printv("Parsing parameters...")

    try:
        json_file = args[0]
    except:
        raise Exception("Must provide an input JSON file")

    if options is None:
        options = {"gpu_accelerated": False}

    try:
        opts, args_left = getopt.getopt(args,"hg", \
                              ["iterations=", "mode=", "bkg_ref=", "illum_ref=", "poisson=", "res_comp=", "illum_mask", "det_mask", \
                               "crop=", "ranks_gpu=", "gpu_weight=","debug", "iter_output=", "save_cxi", "hypers_samples=", "hypers_iter=",\
                               "hypers_share_illum", "hypers_share_backg", "out_of_focus_distance=", "save_frames", "single_overlap"])

    except getopt.GetoptError:
        printv(help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            printv(help)
            sys.exit()
        elif opt in ("-g", "--gpu_accelerated"):
            options["gpu_accelerated"] = True


    if len(args_left) is not 1:

        printv(help)
        sys.exit(2)

    else:
        options["fname"] = args_left[0]

    return options


    
