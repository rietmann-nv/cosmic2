import numpy as np
import xcale
import xcale.common as ptycommon
from xcale.common.misc import imsave_complex, imsave, colorize_complex, get_results_dir
import sys
import os
from PIL import Image

#We would run this guy as:
# python generate_raw_cxi.py /data/smarchesini/newlens_20/200220/NS_200220033/NS_200220033_002.cxi /data/smarchesini/newlens_20/200220/NS_200220033/200220033/001/ /data/smarchesini/newlens_20/200220/NS_200220033/200220033/002/ 2

if __name__ == '__main__':

    args = sys.argv[1:]

    #/data/smarchesini/newlens_20/200220/NS_200220033/NS_200220033_002.cxi This is the cxi pre-processed
    #/data/smarchesini/newlens_20/200220/NS_200220033/200220033/001/ This is the folder of the dark frames
    #/data/smarchesini/newlens_20/200220/NS_200220033/200220033/002/ This is the folder of the raw frames

    cxi_name = args[0] #this contains a pre-processed cxi, we get the metadata from here
    dark_dir_name = args[1] #this is the directory of raw data, where the exp 1, 2 and dark frames would be readed
    raw_dir_name = args[2] #this is the directory of raw data, where the exp 1, 2 and dark frames would be readed
    n_exposures = args[3] #is is either 1 or 2 for single or double exposure

    io = ptycommon.IO()
    metadata = io.read(cxi_name, io.metadataFormat)

    print(metadata)

    dark = []
    raw = []
    lst=os.listdir(dark_dir_name)
    lst.sort()
    
    for fname in lst:
        if not fname.endswith('.tif'): continue
        im = Image.open(os.path.join(dark_dir_name, fname))
        imarray = np.array(im)
        dark.append(imarray)

    dark = np.array(dark)
    print("Dark frames readed, with a size of: " + str(dark.shape))

    lst=os.listdir(raw_dir_name)
    lst.sort()
    
    for fname in lst: #os.listdir(raw_dir_name):
        if not fname.endswith('.tif'): continue
        im = Image.open(os.path.join(raw_dir_name, fname))
        imarray = np.array(im)
        raw.append(imarray)

    raw = np.array(raw)
    print("Raw frames readed, with a size of: " + str(raw.shape))

    data_dictionary = {}

    data_dictionary.update({"raw_data" : raw})
    data_dictionary.update({"dark_data" : dark})

    metadata.update({"n_exposures" : n_exposures})

    output_filename = "raw_" + os.path.basename(cxi_name)

    print("Saving cxi file: " + output_filename)

    #This script deletes and rewrites a previous file with the same name
    try:
        os.remove(output_filename)

    except OSError:
        pass

    io.write(output_filename, metadata, data_format = io.metadataFormat)
    io.write(output_filename, data_dictionary, data_format = io.dataFormat)

