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
        options = {"gpu_accelerated": False, "limit_num_images": None}

    try:
        opts, args_left = getopt.getopt(args,"hgl:", \
                              ["gpu_accelerated", "limit_num_images="])

    except getopt.GetoptError:
        printv(help)
        sys.exit(2)

    print("opts=", opts)
    for opt, arg in opts:
        if opt == '-h':
            printv(help)
            sys.exit()
        elif opt in ("-g", "--gpu_accelerated"):
            options["gpu_accelerated"] = True
        if opt in ("-l", "--limit_num_images"):
            options["limit_num_images"] = int(arg)


    if len(args_left) is not 1:

        printv(help)
        sys.exit(2)

    else:
        options["fname"] = args_left[0]

    print("options=", options)
    
    return options


    
