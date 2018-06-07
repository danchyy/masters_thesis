import os

from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from data_loader.ucf_101_data_loader import Ucf101DataLoader
from utils import constants
import numpy as np
import json

def extract_features():

    train_split = os.path.join(constants.UCF_101_DATA_SPLITS, "train01.txt")
    test_split = os.path.join(constants.UCF_101_DATA_SPLITS, "validation01.txt")
    train_target = os.path.join(constants.UCF_101_LSTM_DATA_AUGMENT, "train")
    test_target = os.path.join(constants.UCF_101_LSTM_DATA_AUGMENT, "test")

    generate_longer = False
    augment_data = True
    data_loader = Ucf101DataLoader(config=dict(), train_split=train_split, test_split=test_split,
                                   generate_longer=generate_longer, augment_data=augment_data)

    # initializing model
    pre_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(constants.IMAGE_DIMS[0],
                                                                                constants.IMAGE_DIMS[1], 3))
    x = pre_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=pre_model.input, outputs=x)
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

    train_split_lines = open(train_split).readlines()
    total_length = len(train_split_lines)
    print("Length of train frame list: " + str(total_length))
    index = 0

    if generate_longer:
        target_dim = (constants.LSTM_SEQUENCE_LENGTH_GENERATION, constants.LSTM_FEATURE_SIZE)
    else:
        target_dim = (constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE)

    for curr_video_key, curr_video, label in data_loader.retrieve_train_data_gen():
        index += 1
        if index % 10 == 0:
            print("Progress: %d / %d" % (index, total_length))
        if curr_video_key + "\n" in visited:
            continue
        features = []
        for frame in curr_video:
            # img = image.load_img(frame_path, target_size=constants.IMAGE_DIMS)
            img = frame
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            feature_vector = model.predict(img)
            feature_vector = np.array(feature_vector[0])
            features.append(feature_vector)

        features = np.array(features)
        if features.shape != target_dim:
            print(features.shape)
            with open("WRONG_SHAPES.txt", "a") as out_file:
                out_file.write(curr_video_key + "\n")
        dest_features_path = os.path.join(train_target, curr_video_key + ".npy")
        np.save(dest_features_path, features)

        label_data = dict()
        label_data["class"] = label.strip()
        label_data["features_path"] = dest_features_path
        dest_label_path = os.path.join(train_target, curr_video_key + ".label.json")
        with open(dest_label_path, 'w') as outfile:
            json.dump(label_data, outfile, indent=3)

        visited.append(curr_video_key + "\n")
        open(log_train, "w").writelines(visited)

    test_split_lines = open(test_split).readlines()
    total_length_test = len(test_split_lines)
    print("Length of train frame list: " + str(total_length_test))
    index = 0

    generate_longer = False
    augment_data = False
    data_loader = Ucf101DataLoader(config=dict(), train_split=train_split, test_split=test_split,
                                   generate_longer=generate_longer, augment_data=augment_data)
    if generate_longer:
        target_dim = (constants.LSTM_SEQUENCE_LENGTH_GENERATION, constants.LSTM_FEATURE_SIZE)
    else:
        target_dim = (constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE)

    for curr_video_key, curr_video, label in data_loader.retrieve_test_data_gen(parse_train=True):
        index += 1
        if index % 10 == 0:
            print("Progress: %d / %d" % (index, total_length_test))
        if curr_video_key + "\n" in visited_test:
            continue
        features = []
        for frame in curr_video:
            # img = image.load_img(frame_path, target_size=constants.IMAGE_DIMS)
            img = frame
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            feature_vector = model.predict(img)
            feature_vector = np.array(feature_vector[0])
            features.append(feature_vector)

        features = np.array(features)
        if features.shape != target_dim:
            with open("WRONG_SHAPES.txt", "a") as out_file:
                out_file.write(curr_video_key + "\n")
        dest_features_path = os.path.join(test_target, curr_video_key + ".npy")
        np.save(dest_features_path, features)

        label_data = dict()
        label_data["class"] = label
        label_data["features_path"] = dest_features_path
        dest_label_path = os.path.join(test_target, curr_video_key + ".label.json")
        with open(dest_label_path, 'w') as outfile:
            json.dump(label_data, outfile, indent=3)

        visited_test.append(curr_video_key + "\n")
        open(log_test, "w").writelines(visited_test)


if __name__ == '__main__':
    extract_features()