from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from utils import constants


class SequenceDataGenerator(Sequence):

    def __init__(self, batch_size, split, num_of_classes, should_average=False, should_subsample=False):
        self.batch_size = batch_size
        self.data_dir = constants.UCF_101_EXTRACTED_FEATURES
        self.split = split
        self.file_names = self.get_file_names()
        shuffle(self.file_names)
        self.num_of_classes = num_of_classes
        self.should_average = should_average
        self.should_submsample = should_subsample
        self.indices_subsample = np.arange(0, 50)

    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        features, labels = [], []
        for file_name in curr_file_names:
            with open(file_name) as in_file:
                json_data = json.load(in_file)
            label = json_data["class"]
            updated_label = int(label) - 1  # SO WE CAN MOVE FIRST LABEL TO ZERO
            labels.append(updated_label)
            curr_feature = np.load(json_data["features_path"])
            if self.should_submsample:
                random_indices = np.sort(np.random.choice(self.indices_subsample, size=30, replace=False))
                curr_feature = curr_feature[random_indices]
            if self.should_average:
                curr_feature = np.mean(curr_feature, axis=0)
                curr_feature = np.expand_dims(curr_feature, axis=0)
            features.append(curr_feature)
        features = np.array(features)
        return features, np.array(to_categorical(labels, num_classes=self.num_of_classes))

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        shuffle(self.file_names)

    def get_file_names(self):
        file_names = open(self.split).readlines()
        target_feature_names = []
        for file_name in file_names:
            name = file_name.split(" ")[0]
            class_name, video_name = name.split("/")
            full_name = class_name + "_" + video_name + ".label.json"
            target_feature_names.append(full_name)
        return target_feature_names
