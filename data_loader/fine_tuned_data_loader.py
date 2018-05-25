from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator
from utils import constants
import os

class FineTunedDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.train_datagen = ImageDataGenerator(rotation_range=15,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                rescale=1. / 255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                fill_mode='nearest')
        self.test_datagen = ImageDataGenerator(rescale=1. / 255.)
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
