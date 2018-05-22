import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from data_loader.ucf_101_data_loader import Ucf101DataLoader
from utils import constants
import numpy as np
import json
from time import time

train_split = os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "cleaned_train01.txt")
test_split = os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, "cleaned_test01.txt")
train_target = os.path.join(constants.UCF_101_LSTM_DATA, "train")
test_target = os.path.join(constants.UCF_101_LSTM_DATA, "test")

data_loader = Ucf101DataLoader(config=dict(), train_split=train_split, test_split=test_split)

train_frame_dict, train_labels, test_frame_dict, test_labels = data_loader.retrieve_frames_list_for_splits()

# initializing model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(constants.RESNET_DIMS[0], constants.RESNET_DIMS[1],
                                                                     3))

log_train = "log_train.txt"
if os.path.exists(log_train):
    visited_log = open(log_train, "r").readlines()
    visited = visited_log
else:
    visited = []

log_test = "log_test.txt"
if os.path.exists(log_test):
    visited_log_test = open(log_test, "r").readlines()
    visited_test = visited_log_test
else:
    visited_test = []

print("Length of train frame list: " + str(len(train_frame_dict)))
total_length = len(train_frame_dict)
index = 0

for curr_video_key in train_frame_dict:
    index += 1
    if index % 10 == 0:
        print("Progress: %d / %d" % (index, total_length))
    if curr_video_key + "\n" in visited:
        continue
    curr_video = train_frame_dict[curr_video_key]
    features = []
    for frame_path in curr_video:
        img = image.load_img(frame_path, target_size=constants.RESNET_DIMS)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature_vector = model.predict(img_data)
        feature_vector = np.array(feature_vector[0][0][0])
        features.append(feature_vector)

    features = np.array(features)
    dest_features_path = os.path.join(train_target, curr_video_key + ".npy")
    np.save(dest_features_path, features)

    label = train_labels[curr_video_key]
    label_data = dict()
    label_data["class"] = label.strip()
    label_data["features_path"] = dest_features_path
    dest_label_path = os.path.join(train_target, curr_video_key + ".label.json")
    with open(dest_label_path, 'w') as outfile:
        json.dump(label_data, outfile, indent=3)

    visited.append(curr_video_key + "\n")
    open(log_train, "w").writelines(visited)


print("Length of train frame list: " + str(len(test_frame_dict)))
total_length_test = len(test_frame_dict)
index = 0

for curr_video_key in test_frame_dict:
    index += 1
    if index % 10 == 0:
        print("Progress: %d / %d" % (index, total_length))
    if curr_video_key + "\n" in visited_test:
        continue
    curr_video = test_frame_dict[curr_video_key]
    features = []
    for frame_path in curr_video:
        img = image.load_img(frame_path, target_size=constants.RESNET_DIMS)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature_vector = model.predict(img_data)
        feature_vector = np.array(feature_vector[0][0][0])
        features.append(feature_vector)

    features = np.array(features)
    dest_features_path = os.path.join(test_target, curr_video_key + ".npy")
    np.save(dest_features_path, features)

    label = test_labels[curr_video_key]
    label_data = dict()
    label_data["class"] = label.strip()
    label_data["features_path"] = dest_features_path
    dest_label_path = os.path.join(test_target, curr_video_key + ".label.json")
    with open(dest_label_path, 'w') as outfile:
        json.dump(label_data, outfile, indent=3)

    visited_test.append(curr_video_key + "\n")
    open(log_test, "w").writelines(visited_test)
