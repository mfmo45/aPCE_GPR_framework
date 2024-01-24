"""
Author: Maria Fernanda Morales Oreamuno
Date created: 18/11/22

ToDo: add a .yaml file load/dump class, which are read as dictionaries, and then use the Dict2Class function
"""

import yaml
import numpy as np


class Dict2Class(object):
    """
    Class transforms a dictionary into a class, with each dictionary key turned into a class attribute.
    Parameters:
        my_dict: <dictionary> with at least one key
        to_numpy: <bool> when True, transforms all lists into a numpy array with the same [rows, columns] as the list

    Notes:
        When transforming lists to numpy array and the resulting array is a 2D array, the list should also be 2D,
        e.g. [[row 1, columns], [row2, columns]]
    """
    def __init__(self, my_dict, to_numpy=False):

        for key in my_dict:
            if isinstance(my_dict[key], list):
                my_dict[key] = np.array(my_dict[key])
            setattr(self, key, my_dict[key])


class Configuration(dict):
    def __init__(self, data_dic):
        super(dict, self).__init__()
        for i, elem in enumerate(data_dic):
            self[elem] = data_dic[elem]


def yaml_loader(filepath):
    """
    Loads yaml file
    :param filepath: string with name of yaml file path
    :return:
    """
    with open(filepath, 'r') as file_descriptor:
        data = yaml.full_load(file_descriptor)
    return data


def yaml_dump(filepath, data):
    with open(filepath, 'w') as file_descriptor:
        yaml.dump(data, file_descriptor)


if __name__ == "__main__":
    file_path = "config.yaml"

    data = yaml_loader(file_path)
    general_data = Configuration(data_dic=data['general_data'])
    print(data)
