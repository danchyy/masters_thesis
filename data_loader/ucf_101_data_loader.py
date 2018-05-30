from base.base_data_loader import BaseDataLoader
from utils import constants
from utils.util_script import get_ucf_101_dict
import os
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import cv2


class Ucf101DataLoader(BaseDataLoader):

    def __init__(self, config: dict, train_split, test_split, max_frames=450, sequence_length=40):
        super().__init__(config)
        self.frames_dir = constants.UCF_101_FRAMES_DIR
        self.max_frames = max_frames
        self.sequence_length = sequence_length
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
        label_dict = get_ucf_101_dict()
        for row in self.test_lines:
            class_name, file_name = self.parse_test_split_row(row)
            final_name = class_name + "_" + file_name
            yield final_name, self.load_frames_from_video(class_name, file_name), label_dict[class_name]

    def retrieve_frames_list_for_splits(self):
        """
        Retrieves frame names for all classes and images in train and test data.

        :return: Train frame names, Train Labels, Test frame names, test labels
        """
        train_frames = dict()
        test_frames = dict()
        train_labels, test_labels = dict(), dict()
        counter = 0
        for row in self.train_lines:
            class_name, file_name, label = self.parse_train_split_row(row)
            final_name = class_name + "_" + file_name
            train_frames[final_name] = self.load_frames_from_video(class_name, file_name)
            # train_frames.append(self.load_frames_list(class_name, file_name))
            train_labels[final_name] = label

        label_dict = get_ucf_101_dict()
        counter = 0
        for row in self.test_lines:
            class_name, file_name = self.parse_test_split_row(row)
            final_name = class_name + "_" + file_name
            test_frames[final_name] = self.load_frames_from_video(class_name, file_name)
            test_labels[final_name] = label_dict[class_name]

            counter += 1
            if counter == 10:
                break

        return train_frames, train_labels, test_frames, test_labels

    def get_length_of_video(self, capture):
        count = 0
        while (True):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if not ret:
                break

            count += 1
        return count

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

            if count in indices_for_sequence:
                image = cv2.resize(image, constants.IMAGE_DIMS)
                frames.append(image)
            success, image = cap.read()
            # print('Read a new frame: ', success)
            count += 1

        frames = frames[:self.sequence_length]
        return frames

    def load_frames_list(self, class_name, file_name):
        frames_full_path = os.path.join(self.frames_dir, class_name, file_name)
        frame_names = os.listdir(frames_full_path)
        frames = []
        frame_ordered_dict = dict()
        for frame_name in frame_names:
            index = int(frame_name.split("_")[1].split(".")[0])
            frame_path = os.path.join(frames_full_path, frame_name)
            frame_ordered_dict[index] = frame_path
        video_length = len(frame_ordered_dict)
        get_every_n = video_length // self.sequence_length
        for key in sorted(frame_ordered_dict):
            frames.append(frame_ordered_dict[key])
        shortened_list = frames[0::get_every_n][:self.sequence_length]
        return shortened_list

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass