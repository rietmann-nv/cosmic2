import numpy as np


def filter(raw_data, dark_data, metadata):

    print("Hey we are applying some filtering and descrambling here")
    data = raw_data + dark_data[0]
    return data
