from utils import constants
import os
from shutil import copyfile
import cv2
import numpy as np


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

def get_length_of_video(capture):
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        if not ret:
            break

        count += 1
    return count


def augment_image(img, inter=cv2.INTER_AREA):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (h, w) = img.shape[:2]

    # check to see if the width is None
    target_dim = constants.IMAGE_DIMS[0]
    target_height = target_dim + 30
    ratio = target_height / h
    new_width = int(ratio * w)
    dim = (new_width, target_height)

    # resize the image
    resized = cv2.resize(img, dim, interpolation=inter)


    random_start = np.random.randint(0, new_width - target_dim - 1)
    random_end = random_start + target_dim
    start_height = np.random.randint(0, target_height - target_dim - 1)
    end_height = start_height + target_dim
    assert (random_end - random_start) == target_dim
    assert (end_height - start_height) == target_dim
    cropped_img = resized[start_height: end_height, random_start: random_end]
    assert cropped_img.shape[:2] == constants.IMAGE_DIMS
    # return the resized image

    return cropped_img


def get_mid_frame_of_video(f_p):
    cap = cv2.VideoCapture(f_p)
    length = get_length_of_video(capture=cap)
    cap = cv2.VideoCapture(f_p)
    success, image = cap.read()
    count = 0
    success = True

    mid_index = int(length / 2)
    while success:
        if count == mid_index:
            image = augment_image(image)
            return image
        success, image = cap.read()
        count += 1


target_dir = constants.UCF_101_CNN_DATA_DIR_1
source_dir = constants.UCF_101_DATA_DIR

train_split = open(os.path.join(constants.UCF_101_DATA_SPLITS, "train_val01.txt"), "r").readlines()

test_split = open(os.path.join(constants.UCF_101_DATA_SPLITS, "test01.txt"), "r").readlines()

train_test_dict = create_train_test_dict(train_split, test_split)

for line in train_split:
    line = line.strip().split(" ")[0]
    class_name, video_name = line.split("/")[0], line.split("/")[1]
    video_path = os.path.join(source_dir, class_name, video_name)
    target_dir_path = os.path.join(target_dir, "train", class_name)
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
    full_path = os.path.join(target_dir_path, video_name.split(".")[0] + ".png")
    mid_frame = get_mid_frame_of_video(video_path)
    cv2.imwrite(full_path, mid_frame)

for line in test_split:
    line = line.strip().split(" ")[0]
    class_name, video_name = line.split("/")[0], line.split("/")[1]
    video_path = os.path.join(source_dir, class_name, video_name)
    target_dir_path = os.path.join(target_dir, "test", class_name)
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
    full_path = os.path.join(target_dir_path, video_name.split(".")[0] + ".png")
    mid_frame = get_mid_frame_of_video(video_path)
    cv2.imwrite(full_path, mid_frame)

"""
for class_name in os.listdir(source_dir):
    if class_name.startswith("."):
        continue
    class_path = os.path.join(source_dir, class_name)
    video_names = os.listdir(class_path)
    train_dest_path = os.path.join(target_dir, "train", class_name)
    test_dest_path = os.path.join(target_dir, "test", class_name)
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
"""