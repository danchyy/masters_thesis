from base.base_data_loader import BaseDataLoader
from utils import constants
import os
from data_generators.lstm_data_generator import LSTMDataGenerator

class LSTMDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.trainer.batch_size
        self.train_dir = os.path.join(constants.UCF_101_LSTM_DATA, "train")
        self.test_dir = os.path.join(constants.UCF_101_LSTM_DATA, "test")

    def get_train_data(self):
        return LSTMDataGenerator(self.batch_size, self.train_dir)

    def get_test_data(self):
        return LSTMDataGenerator(self.batch_size, self.test_dir)
