"""
This module calculate pairwise distance for data that is a numpy array of list of string

parameters:
* X : (numpy array(list(string))) a numpy array of list of string
"""

import numpy as np
from itertools import combinations
from distances import jaccard_cy as jaccard

def calculate_pairwise_distance(X):
    # TODO : this function is very slow need improvement
    data_size = X.shape[0]
    precomputed = np.zeros((data_size, data_size))
    iterator = combinations(range(X.shape[0]), 2)

    for i, j in iterator: 
        precomputed[i, j] = jaccard.jaccard_seq(X[i], X[j])  

    # Make symmetric and return
    return precomputed + precomputed.T - np.diag(np.diag(precomputed))