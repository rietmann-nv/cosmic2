import numpy as np
import sys
import os
import json
from PIL import Image
import xcale.common as ptycommon
from xcale.common.misc import printd, printv


def read_metadata(json_file):

    with open(json_file) as f:
        metadata = json.load(f)

    if "translations" in metadata:

        metadata["translations"] = np.array(metadata["translations"])

    return metadata


def read_data(metadata, json_file, my_indexes):


    dark_frames = None
    raw_frames = None

    base_folder = os.path.split(json_file)[:-1][0] + "/"

    if "dark_dir" in metadata:

        printv("\nReading dark frames from disk...\n")

        #by default we could take the full path, but in this case it has an absolute path from PHASIS,
        #which is not good if you move data to other places
        dark_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["dark_dir"])))

    if "exp_dir" in metadata:

        printv("\nReading raw frames from disk...\n")

        raw_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["exp_dir"])), my_indexes)

    return dark_frames, raw_frames


def read_tiffs(directory, my_indexes = None):

    lst=os.listdir(directory)
    lst.sort()

    #We remove from the list anything that is not a tiff
    lst = [item for item in lst if item.endswith('.tif')]

    if my_indexes is not None:
        lst = lst[my_indexes]

    frames = []    

    ii=0
    n_frames=len(lst)

    for fname in lst:       
        ii+=1
        im = Image.open(os.path.join(directory, fname))
        imarray = np.array(im)
        frames.append(imarray)
        sys.stdout.write('\r file = %s/%s ' %(ii,n_frames))
        sys.stdout.flush()
        
    print("\n")
    return np.array(frames)

