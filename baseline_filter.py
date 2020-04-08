import numpy as np
from xcale.common.misc import printv

def allocate(data_shape, xp):

    local_data = {}

    local_data["some_aux_big_structure"] = xp.zeros(data_shape, dtype= xp.float32)
    local_data["maybe_the_output_if_needed"] = xp.zeros(data_shape, dtype= xp.float32)

    return local_data

def initialize(local_data, xp): return local_data

def filter(local_data, input_data, metadata, mode, xp):

    if mode is "python":
        printv("Hey we are applying some filtering on a CPU here.")

    elif mode is "cuda":
        printv("Hey we are applying some filtering on a GPU here.")

    else:
        printv("ERROR: Unsupported mode.")
        return None

    local_data["maybe_the_output_if_needed"] = input_data["raw_data"] + input_data["dark_data"][0] + local_data["some_aux_big_structure"]

    output_data = {"preproc_data": local_data["maybe_the_output_if_needed"]}

    return output_data
