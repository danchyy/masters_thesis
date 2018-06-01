from base.base_data_loader import BaseDataLoader
from utils import constants
import os
from data_generators.sequence_data_generator import SequenceDataGenerator

class DNNDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.trainer.batch_size
        self.train_dir = os.path.join(constants.UCF_101_LSTM_DATA_train01, "train")
        self.test_dir = os.path.join(constants.UCF_101_LSTM_DATA_train01, "test")

    def get_train_data(self):
        return SequenceDataGenerator(self.batch_size, self.train_dir, self.config.exp.num_of_classes)

    def get_test_data(self):
        return SequenceDataGenerator(self.batch_size, self.test_dir, self.config.exp.num_of_classes)
