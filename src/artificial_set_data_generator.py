"""
This module contain the implementation of artificial data generator for set data

parameters:
* data_size : (int) an integer number specifies number of total number of data that will be generated
* size_of_clusters : (numpy arry) specifies size for each cluster. If empty array is passed then the size of all cluster will be the same.
                Note that len of array should equal to number_of_cluster and sum of this array should equal to data_size
* number_of_cluster : (int) an integer number specifies number of cluster to create
* dimension : (int) an integer number specifies total number of features that will be generate in the data set
* distance_threshold : (float) a number specifies the maximum distance away from the cluster representative according to Jaccard's method
* size_of_set : (tuple(int,int)) a tuple of intergers specifies the minimum and maximum feature that each data has to contain
* all_features : (string[]) an array of string containing all possible features of the dataset

returns:
type : tuple
* artificially generated data set as a list of numpy array containing strings
* ground truth of the data set
* list of numpy array (cluster representative / medoids)
"""
from entries import entry
from distances import overlap

def generate(
    data_size, 
    size_of_clusters, 
    number_of_cluster, 
    dimension, 
    distance_threshold, 
    size_of_set,
    all_features):

    representatives = entry.create_cluster_representatives(
        number_of_cluster, 
        size_of_set, 
        all_features)
    print('=== done representative calculation ===')

    data, ground_truths = entry.generate_cluster_members(
        data_size, 
        representatives, 
        size_of_set, 
        all_features,
        distance_threshold,
        size_of_clusters)
    print('=== done generating data and ground truths ===')

    overlap.calculate_overlap(data, ground_truths, representatives)
    print('=== done overlap calculation ===')

    return(data, ground_truths, representatives)
