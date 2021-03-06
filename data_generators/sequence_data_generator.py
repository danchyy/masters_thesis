from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from utils import constants


class SequenceDataGenerator(Sequence):

    def __init__(self, batch_size, split, num_of_classes, should_average=False, should_subsample=False,
                 load_two_streams=False):
        self.batch_size = batch_size
        self.data_dir = constants.UCF_101_EXTRACTED_FEATURES
        self.split = split
        self.file_names = self.get_file_names()
        shuffle(self.file_names)
        self.num_of_classes = num_of_classes
        self.should_average = should_average
        self.should_submsample = should_subsample
        self.indices_subsample = np.arange(0, 50)
        self.load_two_streams = load_two_streams
        self.opt_flow_dir = constants.UCF_101_EXTRACTED_FLOW
        self.flow_file_names = self.get_file_names(is_flow=load_two_streams)

    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        curr_flow_file_names = self.flow_file_names[start_index : end_index]
        features, labels = [], []
        flow_features = []
        for index, file_name in enumerate(curr_file_names):
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
            if self.load_two_streams:
                with open(curr_flow_file_names[index]) as f:
                    json_data = json.load(f)
                    flow_features.append(np.load(json_data["features_path"]))
            features.append(curr_feature)
        features = np.array(features)
        if self.load_two_streams:
            flow_features = np.array(flow_features)
            return [features, flow_features], np.array(to_categorical(labels, num_classes=self.num_of_classes))
        return features, np.array(to_categorical(labels, num_classes=self.num_of_classes))

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.load_two_streams:
            concated = list(zip(self.file_names, self.flow_file_names))
            shuffle(concated)
            self.file_names, self.flow_file_names = zip(*concated)
        else:
            shuffle(self.file_names)

    def get_file_names(self, is_flow=False):
        file_names = open(self.split).readlines()
        target_feature_names = []
        for file_name in file_names:
            file_name = file_name.strip()
            name = file_name.split(" ")[0]
            class_name, video_name = name.split("/")
            full_name = class_name + "_" + video_name + ".label.json"
            if is_flow:
                target_feature_names.append(os.path.join(self.opt_flow_dir, full_name))
            else:
                target_feature_names.append(os.path.join(self.data_dir, full_name))
        return target_feature_names
