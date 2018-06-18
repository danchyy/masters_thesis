from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator
from utils import constants
from keras.applications.inception_v3 import preprocess_input
import os

class FineTunedDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.train_datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=0.5,
                                                shear_range=0.2, zoom_range=0.2,
                                                preprocessing_function=preprocess_input)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_split = self.config.trainer.train_split
        data_dir = constants.UCF_101_CNN_DATA_DIR_1
        if train_split == "train_val02.txt":
            data_dir = constants.UCF_101_CNN_DATA_DIR_2
        if train_split == "train_val03.txt":
            data_dir = constants.UCF_101_CNN_DATA_DIR_3
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")

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
