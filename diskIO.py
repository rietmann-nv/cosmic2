import numpy as np
import sys
import os
import json
from PIL import Image
import xcale.common as ptycommon


def read(json_file):

    fname_no_extension = os.path.splitext(json_file)[:-1][0]
    cxi_fname = fname_no_extension + "_raw.cxi"

    print(cxi_fname)

    #Check if there is a CXI with the raw data, if so we use it because it is faster
    #if os.path.exists(cxi_fname):

    #    print("CXI file exists, reading raw data from it.")

    #    io = ptycommon.IO()
    #    metadata = io.read(cxi_fname, io.metadataFormat) #This guy has all information needed

    #    raw_frames = io.read(cxi_fname, {"raw_data": ptycommon.IO().dataFormat["raw_data"]})
    #    dark_frames = io.read(cxi_fname, {"dark_data": ptycommon.IO().dataFormat["dark_data"]})

    #else:

    #    print("No raw CXI file exists, reading raw data from the tiff files.")

    metadata, dark_frames, raw_frames = read_from_json(json_file)

        #we also generate a CXI file, to speed up reading next time we use this dataset
        #write_raw_CXI(json_file, metadata, dark_frames, raw_frames)

    return metadata, dark_frames, raw_frames

def read_from_json(json_file):

    metadata = {}
    dark_frames = None
    raw_frames = None

    with open(json_file) as f:
        metadata = json.load(f)

    print(metadata)

    base_folder = os.path.split(json_file)[:-1][0] + "/"

    print(os.path.split(json_file)[1])
    print(os.path.split(json_file))
    print(base_folder)

    if "dark_dir" in metadata:

        #by default we could take the full path, but in this case it has an absolute path from PHASIS,
        #which is not good if you move data to other places
        dark_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["dark_dir"])))

    if "exp_dir" in metadata:

        raw_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["exp_dir"])))

    if "translations" in metadata:

        metadata["translations"] = np.array(metadata["translations"])

    return metadata, dark_frames, raw_frames


def read_tiffs(directory):

    lst=os.listdir(directory)
    lst.sort()

    frames = []    

    ii=0
    n_frames=len(lst)

    for fname in lst:       
        ii+=1
        if not fname.endswith('.tif'): continue
        im = Image.open(os.path.join(directory, fname))
        imarray = np.array(im)
        frames.append(imarray)
        sys.stdout.write('\r file = %s/%s ' %(ii,n_frames))
        sys.stdout.flush()
        
    print("\n")
    return np.array(frames)


def write_raw_CXI(file_name, metadata, dark_frames, raw_frames):

    io = ptycommon.IO()

    data_dictionary = {}

    data_dictionary.update({"raw_data" : raw_frames})
    data_dictionary.update({"dark_data" : dark_frames})

    output_filename = os.path.splitext(file_name)[:-1][0] + "_raw.cxi"

    print("Saving cxi file: " + output_filename)

    #This script deletes and rewrites a previous file with the same name
    try:
        os.remove(output_filename)

    except OSError:
        pass

    io.write(output_filename, metadata, data_format = io.metadataFormat)
    io.write(output_filename, data_dictionary, data_format = io.dataFormat)

    print("File saved: " + output_filename)


