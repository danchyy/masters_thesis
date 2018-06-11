from utils import constants
import os
from shutil import copyfile

root_dir = constants.UCF_101_EXTRACTED_FEATURES
test_split = os.path.join(constants.UCF_101_DATA_SPLITS, "test01.txt")

target_dir = constants.UCF_101_EXTRACTED_FEATURES_TEST_1

test_lines = open(test_split).readlines()

for line in test_lines:
    line = line.strip()
    splitted_line = line.split("/")

    label_file_name = splitted_line[0] + "_" + splitted_line[1] + ".label.json"
    full_source_path = os.path.join(root_dir, label_file_name)

    full_target_path = os.path.join(target_dir, label_file_name)
    copyfile(full_source_path, full_target_path)