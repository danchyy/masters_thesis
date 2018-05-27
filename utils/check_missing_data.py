import os
import constants

train_split = os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "cleaned_train01.txt")
test_split = os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "cleaned_test01.txt")

target_video_dir = constants.UCF_101_DATA_DIR

train_lines = open(train_split).readlines()
test_lines = open(test_split).readlines()

for line in train_lines:
    video_path = line.split(" ")[0]
    full_video_path = os.path.join(target_video_dir, video_path)
    if not os.path.exists(full_video_path):
        print(full_video_path)

for line in test_lines:
    line = line.strip()
    full_video_path = os.path.join(target_video_dir, line)
    if not os.path.exists(full_video_path):
        print(full_video_path)
