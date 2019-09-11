"""
This module and read and manipulate the raw data

APIs 
* read_file : read from file and create numpy array of list
    * parameters
        - file_path : (string) file path to supply to data_utilities

* get_features : flatten strings from numpy array of list and unique them
    * parameters
        - np_data : numpy array of list
"""

import numpy as np
from itertools import chain

def read_file(file_path):
    data_list = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            line = line.split(' ')
            data_list.append(line)

    return np.array(data_list)

def get_features(np_data):
    np_data_1d = list(chain.from_iterable(np_data))
    unique_set = set(np_data_1d)
    unique_set.discard('')
    features = np.array(list(unique_set))
    return features