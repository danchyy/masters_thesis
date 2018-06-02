from base.base_data_loader import BaseDataLoader
from utils import constants
from utils.util_script import get_ucf_101_dict
import os
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import cv2


class Ucf101DataLoader(BaseDataLoader):

    def __init__(self, config: dict, train_split, test_split):
        super().__init__(config)
        self.frames_dir = constants.UCF_101_FRAMES_DIR
        self.train_lines = open(train_split).readlines()
        self.test_lines = open(test_split).readlines()
        self.resnet_dims = constants.IMAGE_DIMS

    def parse_train_split_row(self, train_row):
        label = train_row.split(" ")[1]
        name_only = train_row.split(" ")[0]
        # class_name, file_name = name_only.split("/")[0], name_only.split("/")[1].split(".")[0]
        class_name, file_name = name_only.split("/")[0], name_only.split("/")[1]  # version for video
        return class_name, file_name, label

    def parse_test_split_row(self, test_row):
        # class_name, file_name = test_row.split("/")[0], test_row.split("/")[1].split(".")[0]
        class_name, file_name = test_row.split("/")[0], test_row.split("/")[1].strip()  # version for video
        return class_name, file_name

    def retrieve_train_data_gen(self):
        """
        Creates generator for train data
        :return: Name of file, list of frames, label
        """
        for row in self.train_lines:
            class_name, file_name, label = self.parse_train_split_row(row)
            final_name = class_name + "_" + file_name
            yield final_name, self.load_frames_from_video(class_name, file_name), label

    def retrieve_test_data_gen(self):
        """
        Creates generator for test data
        :return: Name of file, list of frames, label
        """
        for row in self.test_lines:
            class_name, file_name, label = self.parse_train_split_row(row)
            final_name = class_name + "_" + file_name
            yield final_name, self.load_frames_from_video(class_name, file_name), label

    def get_length_of_video(self, capture):
        count = 0
        while (True):
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

        """if np.random.rand() < 0.3:
            cropped_img = cv2.flip(cropped_img, 0)  # horizontal flip
        if np.random.rand() < 0.3:
            cropped_img = cv2.flip(cropped_img, 1)  # vertical flip"""


        return cropped_img

    def load_frames_from_video(self, class_name, file_name):
        full_path = os.path.join(constants.UCF_101_DATA_DIR, class_name, file_name)
        cap = cv2.VideoCapture(full_path)
        length = self.get_length_of_video(capture=cap)
        cap = cv2.VideoCapture(full_path)
        frames = []
        success, image = cap.read()
        count = 0
        success = True

        indices_for_sequence = [int(a) for a in np.arange(0, length, length / constants.LSTM_SEQUENCE_LENGTH)]
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if count in indices_for_sequence:
                image = self.augment_image(image)
                frames.append(image)
            success, image = cap.read()
            count += 1

        cap = cv2.VideoCapture(full_path)
        while len(frames) < constants.LSTM_SEQUENCE_LENGTH:
            success, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.augment_image(image)
            frames.append(image)
        frames = frames[:constants.LSTM_SEQUENCE_LENGTH]
        return frames

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass