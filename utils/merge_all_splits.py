"""
Script which merges all three train splits into one and all three test scripts into one so that all files can be
extracted and processed
"""
from utils import constants
import os


train_val_splits = ["train_val01.txt", "train_val02.txt", "train_val03.txt"]

test_splits = ["test01.txt", "test02.txt", "test03.txt"]

all_train_lines = []

root_dir = constants.UCF_101_DATA_SPLITS

train_lines = []

for split in train_val_splits:
    split_path = os.path.join(root_dir, split)
    lines = open(split_path).readlines()
    for line in lines:
        if line not in all_train_lines:
            all_train_lines.append(line)
            train_lines.append(line.split(" ")[0].strip())

all_train_lines.sort()
open(os.path.join(root_dir, "all_train_lines.txt"), "w").writelines(all_train_lines)

all_test_lines = []

for split in test_splits:
    split_path = os.path.join(root_dir, split)
    lines = open(split_path).readlines()
    for line in lines:
        if line not in all_test_lines:
            all_test_lines.append(line)

all_test_lines.sort()
open(os.path.join(root_dir, "all_test_lines.txt"), "w").writelines(all_test_lines)