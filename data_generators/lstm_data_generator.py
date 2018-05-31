from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from utils import constants
import cv2


class LSTMDataGenerator(Sequence):

    def __init__(self, batch_size, split, num_of_classes, is_train=True):
        self.batch_size = batch_size
        self.split = split
        self.file_names = open(os.path.join(constants.UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR, self.split)).readlines()
        self.num_of_classes = num_of_classes
        if is_train:
            self.image_datagen = ImageDataGenerator(rotation_range=15,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    rescale=1. / 255,
                                                    shear_range=0.2,
                                                    zoom_range=0.2,
                                                    fill_mode='nearest')
        else:
            self.image_datagen = ImageDataGenerator(rescale=1. / 255.)
        self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=(constants.IMAGE_DIMS[0],
                                                                                     constants.IMAGE_DIMS[1], 3))
        self.model._make_predict_function()
        self.is_train = is_train

    def get_length_of_video(self, capture):
        count = 0
        while True:
            # Capture frame-by-frame
            ret, frame = capture.read()
            if not ret:
                break

            count += 1
        return count

    def augment_image(self, img, inter=cv2.INTER_AREA):
        (h, w) = img.shape[:2]

        # check to see if the width is None
        target_height = constants.IMAGE_DIMS[0]
        ratio = target_height / h
        new_width = int(ratio * w)
        dim = (new_width, target_height)

        # resize the image
        resized = cv2.resize(img, dim, interpolation=inter)

        random_start = np.random.randint(0, new_width - target_height - 1)
        random_end = random_start + target_height
        assert (random_end - random_start) == target_height
        cropped_img = resized[:, random_start: random_end]
        assert cropped_img.shape[:2] == constants.IMAGE_DIMS
        # return the resized image

        if np.random.rand() < 0.5:
            cropped_img = cv2.flip(cropped_img, 0)  # horizontal flip
        if np.random.rand() < 0.5:
            cropped_img = cv2.flip(cropped_img, 1)  # vertical flip

        return cropped_img

    def extract_features(self, img):
        resized_img = self.augment_image(img)
        img_data = image.img_to_array(resized_img)
        img_data = np.expand_dims(img_data, axis=0)
        # img_data = self.image_datagen.flow(img_data, batch_size=1)
        img_data = preprocess_input(img_data)
        feature_vector = self.model.predict(img_data)
        feature_vector = np.array(feature_vector[0][0][0])
        return feature_vector

    def load_frames_from_video(self, class_name, file_name):
        full_path = os.path.join(constants.UCF_101_DATA_DIR, class_name, file_name)
        cap = cv2.VideoCapture(full_path)
        length = self.get_length_of_video(capture=cap)
        cap = cv2.VideoCapture(full_path)
        frames = []
        success, img = cap.read()
        count = 0
        success = True

        indices_for_sequence = [int(a) for a in np.arange(0, length, length / constants.LSTM_SEQUENCE_LENGTH)]
        while success:

            if length < constants.LSTM_SEQUENCE_LENGTH and count == 0:
                feature_vector = self.extract_features(img)
                frames.append(feature_vector)

            if count in indices_for_sequence:
                feature_vector = self.extract_features(img)
                frames.append(feature_vector)
            success, img = cap.read()
            # print('Read a new frame: ', success)
            count += 1

        frames = np.array(frames)
        assert frames.shape == (constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE)
        return frames

    def parse_train_split_row(self, train_row):
        label = train_row.split(" ")[1]
        name_only = train_row.split(" ")[0]
        # class_name, file_name = name_only.split("/")[0], name_only.split("/")[1].split(".")[0]
        class_name, file_name = name_only.split("/")[0], name_only.split("/")[1]  # version for video
        return class_name, file_name, label


    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        features, labels = [], []
        for file_name in curr_file_names:

            class_name, file_name, label = self.parse_train_split_row(file_name)
            curr_feature = self.load_frames_from_video(class_name, file_name)
            features.append(curr_feature)
            labels.append(label)
            # with open(file_name) as in_file:
                # json_data = json.load(in_file)
            """label = json_data["class"]
            updated_label = int(label) - 1  # SO WE CAN MOVE FIRST LABEL TO ZERO
            labels.append(updated_label)
            curr_feature = np.load(json_data["features_path"])
            features.append(curr_feature)"""

        return np.array(features), np.array(to_categorical(labels, num_classes=self.num_of_classes))

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        shuffle(self.file_names)
