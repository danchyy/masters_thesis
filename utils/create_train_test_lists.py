"""
This scripts creates train and test splits based on the number of frames. It takes existing splits and removes the files
from the split which have more or less than available frames.
"""
import constants
import os

min_frames = 40
max_frames = 450

split_dir = constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR

train_file_name = "train_val01.txt"
test_file_name = "test01.txt"

frames_dir = constants.UCF_101_FRAMES_DIR

train_lines = open(os.path.join(split_dir, train_file_name)).readlines()
test_lines = open(os.path.join(split_dir, test_file_name)).readlines()

new_lines = []
for line in train_lines:
    name_only = line.split(" ")[0]
    class_name, file_name = name_only.split("/")[0], name_only.split("/")[1].split(".")[0]
    frames_path = os.path.join(frames_dir, class_name, file_name)

    number_of_frames = len(os.listdir(frames_path))
    if number_of_frames < min_frames or number_of_frames > max_frames:
        continue
    new_lines.append(line)

open(os.path.join(split_dir, "cleaned_train01.txt"), "w").writelines(new_lines)

new_lines = []

for line in test_lines:
    class_name, file_name = line.split("/")[0], line.split("/")[1].split(".")[0]
    frames_path = os.path.join(frames_dir, class_name, file_name)

    number_of_frames = len(os.listdir(frames_path))
    if number_of_frames < min_frames or number_of_frames > max_frames:
        continue
    new_lines.append(line)

open(os.path.join(split_dir, "cleaned_test01.txt"), "w").writelines(new_lines)