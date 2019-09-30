from data import data_utilities
import numpy as np

def map_original_to_generic_name(file_path, out_path = '../data/generic_data.txt', lookup_out_path = '../data/lookup_data.txt'):
    ALL_FEATURE_FILE_PATH = file_path

    all_unique_features = data_utilities.get_features(data_utilities.read_file(ALL_FEATURE_FILE_PATH))

    generic_data_names = []
    for i in range(len(all_unique_features)):
        generic_data_names.append('P' + "{:04d}".format(i))

    data_utilities.write_file([generic_data_names], out_path)
    data_utilities.write_file([all_unique_features], lookup_out_path)

def convert_sequence(data_seq, save_path='../data/generic_data.txt', look_up_save_path = '../data/lookup_data.txt'):
    generic_data = data_utilities.read_file(save_path)[0]
    lookup = data_utilities.read_file(look_up_save_path)[0]

    all_converted_seq = []
    for i in range(len(data_seq)):
        each_converted_seq = []
        for j in data_seq[i]:
            index = np.where(lookup == j)[0][0]
            each_converted_seq.append(generic_data[index])

        all_converted_seq.append(np.array(each_converted_seq))

    return all_converted_seq
    



