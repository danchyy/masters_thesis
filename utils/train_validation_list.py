import os
from sklearn.model_selection import train_test_split
from utils import constants
import numpy as np
from collections import Counter
import math

def split_current_class(dict_data, number_of_groups):
    """

    :param dict_data:
    :return: Train lines and test lines
    """
    number_of_validation = round(0.2 * number_of_groups)
    count = 0
    train_lines, validation_lines = [], []
    for key in dict_data:
        if count < number_of_validation:
            validation_lines.extend(dict_data[key])
        else:
            train_lines.extend(dict_data[key])
        count += 1
    return train_lines, validation_lines


train_splits = ["trainlist01.txt", "trainlist02.txt", "trainlist03.txt"]

for split in train_splits:
    lines = open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, split)).readlines()
    x, y = [], []
    counter = Counter()
    number_of_groups = 1
    previous_class_name, previous_group = None, None
    dict_data = {}
    all_train, all_validation = [], []
    for line in lines:
        splitted_line = line.split(" ")
        file_name = splitted_line[0]
        class_name = file_name.split("/")[0]

        stripped_underscores = line.split("_")
        group = stripped_underscores[2]
        if previous_class_name is None:
            previous_class_name = class_name
            previous_group = group
        if class_name != previous_class_name:
            train_lines, validation_lines = split_current_class(dict_data, number_of_groups)
            all_train.extend(train_lines)
            all_validation.extend(validation_lines)
            dict_data.clear()
            number_of_groups = 1
            previous_group = group
            previous_class_name = class_name

        if group not in dict_data:
            dict_data[group] = [line]
        else:
            dict_data[group].append(line)
        if previous_group != group:
            previous_group = group
            number_of_groups += 1

    train_lines, validation_lines = split_current_class(dict_data, number_of_groups)
    all_train.extend(train_lines)
    all_validation.extend(validation_lines)
    file_index = split.split(".")[0][-2:]
    train_name = "train" + file_index + ".txt"
    validation_name = "validation" + file_index + ".txt"
    print("Writing to: " + train_name)
    open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, train_name), "w").writelines(all_train)
    print("Writing to: " + validation_name)
    open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, validation_name), "w").writelines(all_validation)
    for line in all_validation:
        if line in all_train:
            print("ERROR")
