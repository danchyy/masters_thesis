from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle
from keras.utils import to_categorical


class SequenceDataGenerator(Sequence):

    def __init__(self, batch_size, data_dir, num_of_classes, should_average=False):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.file_names = self.get_names_in_dir()
        self.num_of_classes = num_of_classes
        self.should_average = should_average

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

    def get_names_in_dir(self):
        file_names = os.listdir(self.data_dir)
        file_names = [os.path.join(self.data_dir, file_name) for file_name in file_names if file_name.endswith(".json")]
        return file_names
