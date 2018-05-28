from keras.preprocessing.image import ImageDataGenerator

from base.base_data_loader import BaseDataLoader
from utils import constants
import os

class LSTMDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)


    def get_train_data(self):
        pass

    def get_test_data(self):
        pass