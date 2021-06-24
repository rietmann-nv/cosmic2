#!/usr/bin/env python

import sys, getopt, os
from cosmicp.common import printv, color, bcolors

default_conf = os.path.join(os.path.expanduser('~')) + "/cosmicp_config/default.json" #This default.json is written there during installation

help =   "\nUsage: cosmicp.py [options] input.json\n\n\
\t -g   -> Perform a GPU execution, off by default.\n\
\t -c F -> Using a configuration file F. If not given the default configuration is pulled from {}.\n\
\t -b N -> Set local batch size = N, per MPI rank. N = 20 by default.\n\
\n\n".format(default_conf)

def parse_arguments(args, options = None):

    printv(color("\nParsing parameters...\n", bcolors.OKGREEN))

    try:
        json_file = args[0]
    except:
        raise Exception(color("\nMust provide an input JSON file\n", bcolors.FAIL))

    if options is None:
        options = {"gpu_accelerated": False,
                   "conf_file":  default_conf,
                   "batch_size_per_rank": 20}

    try:
        opts, args_left = getopt.getopt(args,"hgc:b:", \
                              ["gpu_accelerated", "conf_file=", "batch_size_per_rank="])

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
        if opt in ("-c", "--conf_file"):
            options["conf_file"] = str(arg)


    if len(args_left) != 1:

        printv(color(help, bcolors.WARNING))
        sys.exit(2)

    else:
        options["fname"] = args_left[0]
    
    return options


    
