from base.base_data_loader import BaseDataLoader
from utils import constants
from utils.util_script import get_ucf_101_dict
import os
from typing import List, Tuple
from keras.preprocessing import image
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
        self.resnet_dims = constants.RESNET_DIMS

    def parse_train_split_row(self, train_row):
        label = train_row.split(" ")[1]
        name_only = train_row.split(" ")[0]
        class_name, file_name = name_only.split("/")[0], name_only.split("/")[1].split(".")[0]
        return class_name, file_name, label

    def parse_test_split_row(self, test_row):
        class_name, file_name = test_row.split("/")[0], test_row.split("/")[1].split(".")[0]
        return class_name, file_name

    def retrieve_frames_list_for_splits(self):
        """
        Retrieves frame names for all classes and images in train and test data.

        :return: Train frame names, Train Labels, Test frame names, test labels
        """
        train_frames = []
        test_frames = []
        train_labels, test_labels = [], []
        for row in self.train_lines:
            class_name, file_name, label = self.parse_train_split_row(row)
            train_frames.append(self.load_frames_list(class_name, file_name))
            train_labels.append(label)

        label_dict = get_ucf_101_dict()
        for row in self.test_lines:
            class_name, file_name = self.parse_test_split_row(row)
            test_frames.append(self.load_frames_list(class_name, file_name))
            test_labels.append(label_dict[class_name])

        return train_frames, train_labels, test_frames, test_labels

    def load_frames_list(self, class_name, file_name):
        frames_full_path = os.path.join(self.frames_dir, class_name, file_name)
        frame_names = os.listdir(frames_full_path)
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(frames_full_path, frame_name)
            frames.append(frame_path)
        video_length = len(frames)
        get_every_n = video_length // self.sequence_length
        return frames[0::get_every_n][:self.sequence_length]

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass