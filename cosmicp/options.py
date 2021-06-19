#!/usr/bin/env python

import sys, getopt
from cosmicp.common import printv, color, bcolors


help =   "\nUsage: cosmicp.py [options] input.json\n\n\
\t -g   -> Perform a GPU execution, off by default.\n\
\t -b N -> Set local batch size = N, per MPI rank. N = 20 by default.\n\
\n\n"

def parse_arguments(args, options = None):

    printv(color("\nParsing parameters...\n", bcolors.OKGREEN))

    try:
        json_file = args[0]
    except:
        raise Exception(color("\nMust provide an input JSON file\n", bcolors.FAIL))

    if options is None:
        options = {"gpu_accelerated": False, 
                   "batch_size_per_rank": 20}

    try:
        opts, args_left = getopt.getopt(args,"hgb:", \
                              ["gpu_accelerated", "batch_size_per_rank="])

    except getopt.GetoptError:
        printv(color(help, bcolors.WARNING))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            printv(help)
            sys.exit()
        elif opt in ("-g", "--gpu_accelerated"):
            options["gpu_accelerated"] = True
        if opt in ("-b", "--batch_size_per_rank"):
            options["batch_size_per_rank"] = int(arg)


    if len(args_left) != 1:

        printv(color(help, bcolors.WARNING))
        sys.exit(2)

    else:
        options["fname"] = args_left[0]
    
    return options


    
