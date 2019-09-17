"""
This module contains APIs for creating the data entry with different criteria

APIs 
* create_cluster_representatives : create a data entry with ramdom member to be used as cluster medoids
    * parameters
        - number_of_cluster : (int) an integer number specifies number of cluster to create
        - size_of_set : (tuple(int,int)) a tuple of intergers specifies the minimum and maximum feature that each data has to contain
        - all_features : (string[]) an array of string containing all possible features of the dataset
    * return
        - all_representative : list of numpy array (cluster representative / medoids)

* generate_cluster_members : create data entries for all cluster along with its ground truth
    * parameters
        - data_size : (int) an integer number specifies number of total number of data that will be generated
        - representatives : (list(numpy array)) list of numpy array (cluster representative / medoids)
        - size_of_set : (tuple(int,int)) a tuple of intergers specifies the minimum and maximum feature that each data has to contain
        - all_features : (string[]) an array of string containing all possible features of the dataset
        - distance_threshold : (float) a number specifies the maximum distance away from the cluster representative according to Jaccard's method
    * return
        type : tuple
        - artificially generated data set as a list of numpy array containing strings
        - ground truth of the data set
"""

import numpy as np
import random
import math
from distances import jaccard

DEFAULT_STRING = '   '

def _create_cluster_member_random(all_features, min_feature, max_feature):
    number_of_member_features = random.randrange(min_feature, max_feature + 1)
    return random.sample(all_features.tolist(), number_of_member_features)

def _find_number_of_member_per_cluster(data_size, representatives):
    n_centers = len(representatives)
    number_of_data_per_cluster = [int((data_size) // n_centers)] * n_centers

    for i in range(data_size % n_centers):
        number_of_data_per_cluster[i] += 1

    return number_of_data_per_cluster

# [Unused]
def _create_cluster_member_with_one_different(representative, all_features):
    cp_representative = np.copy(representative)
    feature_id_to_change = random.randint(1, len(representative) - 1)
    new_feature = random.sample(all_features.tolist(), 1)[0]

    cp_representative[feature_id_to_change] = new_feature

    return cp_representative

# [Unused]
def _create_cluster_member_random_but_same_feature_length(representative, all_features):
    cp_representative = np.copy(representative)

    # How many number of ones in each sequence
    selected_features = np.random.randint(0, 2, size=len(representative))
    feature_ids_to_change = np.where(selected_features == 1)[0]

    if feature_ids_to_change.shape[0] == 0:
        return _create_cluster_member_with_one_different(representative, all_features)
    else:
        new_features = random.sample(all_features.tolist(), feature_ids_to_change.shape[0])[0]
        cp_representative[feature_ids_to_change] = new_features

        return cp_representative

def _create_cluster_member_random_different_feature_length(
    representative, 
    all_features, 
    min_feature, 
    max_feature,
    distance_threshold):
    similarity_threshold = 1 - distance_threshold

    number_of_member_features = random.randrange(min_feature, max_feature + 1)
    min_intersect = math.floor((similarity_threshold * (len(representative) + number_of_member_features)) / (1 + similarity_threshold))

    while(min_intersect >= number_of_member_features or min_intersect >= len(representative)):
        number_of_member_features = random.randrange(min_feature, max_feature + 1)
        min_intersect = math.floor((similarity_threshold * (len(representative) + number_of_member_features)) / (1 + similarity_threshold))

    new_member = np.full(number_of_member_features, DEFAULT_STRING)
    number_of_intersect = np.random.randint(min_intersect, min(number_of_member_features, len(representative)))
    total_available_features = [x for x in all_features if x not in representative]
    
    if number_of_member_features > len(representative):
        new_member[0:len(representative)] = representative
        id_to_change = random.sample(range(len(representative)), len(representative) - number_of_intersect)
        empty_space = len(np.where(new_member == DEFAULT_STRING)[0])
        new_features_pool = random.sample(total_available_features, len(id_to_change) + empty_space)
        new_member[len(representative):] = new_features_pool[0:(number_of_member_features - len(representative))]
        new_member[id_to_change] = new_features_pool[(number_of_member_features - len(representative)):]
    elif number_of_member_features < len(representative):
        selected_id = random.sample(range(len(representative)), number_of_member_features)
        new_member = representative[selected_id]
        new_features = random.sample(total_available_features, len(new_member) - number_of_intersect)
        id_to_change = random.sample(range(len(new_member)), len(new_member) - number_of_intersect)
        new_member[id_to_change] = new_features
    else:
        new_member = np.copy(representative)
        id_to_change = random.sample(range(number_of_member_features), number_of_member_features - number_of_intersect)
        new_features = random.sample(total_available_features, len(id_to_change))
        new_member[id_to_change] = new_features
    return new_member

def create_cluster_representatives(number_of_cluster, size_of_set, all_features):    
    all_representative = []
    for i in range(number_of_cluster):
        representative = np.array(_create_cluster_member_random(all_features, size_of_set[0], size_of_set[1]))
        all_representative.append(representative)

    return all_representative

def generate_cluster_members(
    data_size, 
    representatives, 
    size_of_set,
    all_features,
    distance_threshold,
    size_of_clusters):

    if len(size_of_clusters) == 0:
        number_of_data_per_cluster = _find_number_of_member_per_cluster(data_size, representatives)
    else : number_of_data_per_cluster = size_of_clusters

    data = []
    ground_truths = []

    for i in range(len(representatives)):
        cluster_len = 0

        data.append(representatives[i])
        ground_truths.append(i)

        cluster_len = 1

        while cluster_len != number_of_data_per_cluster[i]:
            member = _create_cluster_member_random_different_feature_length(
                representatives[i], 
                all_features, 
                size_of_set[0], 
                size_of_set[1], 
                distance_threshold)

            if jaccard.jaccard_seq(list(representatives[i]), list(member)) < distance_threshold:
                np.random.shuffle(member)
                data.append(member)
                ground_truths.append(i)
                cluster_len = cluster_len + 1

    if len(np.array(ground_truths)) != len(data):
        raise Exception('Program terminated, lengths of data and ground truths are not equal')

    return (data, np.array(ground_truths))
