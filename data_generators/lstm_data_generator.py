from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle


class LSTMDataGenerator(Sequence):

    def __init__(self, batch_size, data_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.file_names = self.get_names_in_dir()

    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        features, labels = [], []
        for file_name in curr_file_names:
            with open(file_name) as in_file:
                json_data = json.load(in_file)
                labels.append(json_data["class"])
                curr_feature = np.load(json_data["features_path"])
                features.append(curr_feature)
        return np.array(features), np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        shuffle(self.file_names)

    def get_names_in_dir(self):
        file_names = os.listdir(self.data_dir)
        file_names = [os.path.join(self.data_dir, file_name) for file_name in file_names if file_name.endswith(".json")]
        return file_names
