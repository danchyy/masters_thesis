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
    class_lines = open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR,
                                    constants.UCF_101_SPLIT_FILE_NAME)).readlines()
    class_dict = dict()
    for line in class_lines:
        splitted_line = line.split(" ")
        class_dict[splitted_line[1]] = int(splitted_line[0])
    return class_dict
