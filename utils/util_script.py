import argparse
import os
from utils import constants


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_ucf_101_dict():
    class_lines = open(os.path.join(constants.UCF_101_DATA_SPLITS,
                                    constants.UCF_101_CLASS_FILE_NAME)).readlines()
    class_dict = dict()
    for line in class_lines:
        splitted_line = line.split(" ")
        class_dict[splitted_line[1].strip()] = int(splitted_line[0])
    return class_dict

def get_ucf_101_label_dict():
    """

    :return: Dict with key->value as label->class
    """
    class_lines = open(os.path.join(constants.UCF_101_DATA_SPLITS,
                                    constants.UCF_101_CLASS_FILE_NAME)).readlines()
    class_dict = dict()
    for line in class_lines:
        splitted_line = line.split(" ")
        class_dict[int(splitted_line[0])] = splitted_line[1].strip()
    return class_dict


def get_number_of_classes():
    class_lines = open(os.path.join(constants.UCF_101_DATA_SPLITS,
                                    constants.UCF_101_CLASS_FILE_NAME)).readlines()
    return len(class_lines)


def get_number_of_items(name_of_split):
    return len(open(os.path.join(constants.UCF_101_DATA_SPLITS, name_of_split)).readlines()) - 1
