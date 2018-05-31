from utils import constants
import os
from shutil import copyfile


def create_train_test_dict(train_split_lines, test_split_lines):
    train_test_dict = dict()
    for line in train_split_lines:
        line = line.split(" ")[0]
        video_name = line.split("/")[1].split(".")[0]
        train_test_dict[video_name] = "TRAIN"
    for line in test_split_lines:
        video_name = line.split("/")[1].split(".")[0]
        train_test_dict[video_name] = "TEST"

    return train_test_dict


target_dir = constants.UCF_101_CNN_DATA_DIR_TRAINLIST01
source_dir = constants.UCF_101_FRAMES_DIR

train_split = open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "train01.txt"), "r").readlines()

test_split = open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "validation01.txt"), "r").readlines()

train_test_dict = create_train_test_dict(train_split, test_split)

for class_name in os.listdir(source_dir):
    if class_name.startswith("."):
        continue
    class_path = os.path.join(source_dir, class_name)
    video_names = os.listdir(class_path)
    train_dest_path = os.path.join(target_dir, "train", class_name)
    test_dest_path = os.path.join(target_dir, "validation", class_name)
    if not os.path.exists(train_dest_path):
        os.makedirs(train_dest_path)
        os.makedirs(test_dest_path)
    for video_name in video_names:
        if video_name.startswith("."):
            continue
        if video_name not in train_test_dict:
            continue
        video_name_path = os.path.join(class_path, video_name)
        frame_in_middle = len(os.listdir(video_name_path)) // 2
        frame_path = os.path.join(video_name_path, "frame_" + str(frame_in_middle) + ".jpg")
        if train_test_dict[video_name] == "TRAIN":
            copyfile(frame_path,
                     os.path.join(target_dir, "train", class_name, video_name + "frame_" + str(frame_in_middle)
                                  + ".jpg"))
        elif train_test_dict[video_name] == "TEST":
            copyfile(frame_path,
                     os.path.join(target_dir, "validation", class_name, video_name + "_frame_" + str(frame_in_middle)
                                  + ".jpg"))
