from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator
from utils import constants
from keras.applications.inception_v3 import preprocess_input
import os

class FineTunedDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_dir = os.path.join(constants.UCF_101_CNN_DATA_DIR, "train")
        self.test_dir = os.path.join(constants.UCF_101_CNN_DATA_DIR, "validation")

    def get_train_data(self):
        return self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=constants.IMAGE_DIMS,
            batch_size=self.config.trainer.batch_size,
            shuffle=True,
            class_mode='categorical')

    def get_test_data(self):
        return self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=constants.IMAGE_DIMS,
            batch_size=self.config.trainer.batch_size,
            shuffle=True,
            class_mode='categorical')
