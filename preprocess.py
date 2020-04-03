import numpy as np
import xcale
import xcale.common as ptycommon
from xcale.common.misc import imsave_complex, imsave, colorize_complex, get_results_dir
import sys
import os
from PIL import Image

from baseline_filter import filter

#We run this guy as:
#python preprocess.py raw_NS_200220033_002.cxi

if __name__ == '__main__':

    args = sys.argv[1:]

    cxi_name = args[0]

    io = ptycommon.IO()
    metadata = io.read(cxi_name, io.metadataFormat) #This guy has all information needed

    n_exposures = metadata["n_exposures"]


    raw_data = io.read(cxi_name, ptycommon.IO().dataFormat)["raw_data"]
    dark_data = io.read(cxi_name, ptycommon.IO().dataFormat)["dark_data"]


    data = filter(raw_data, dark_data, metadata) #This will generate a set of clean frames

    
    data_dictionary = {}
    data_dictionary.update({"data" : data})

    output_filename = "preproc_" + os.path.basename(cxi_name)

    print("Saving cxi file: " + output_filename)

    #This script deletes and rewrites a previous file with the same name
    try:
        os.remove(output_filename)

    except OSError:
        pass

    io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
    io.write(output_filename, data_dictionary, data_format = io.dataFormat) #We use the metadata we readed above and drop it into the new cxi
