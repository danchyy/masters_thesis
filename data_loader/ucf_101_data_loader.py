from base.base_data_loader import BaseDataLoader
from utils import constants
from utils.util_script import get_ucf_101_dict
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import cv2


class Ucf101DataLoader(BaseDataLoader):

    def __init__(self, config: dict, train_split, test_split, generate_longer=False, augment_data=False):
        super().__init__(config)
        self.frames_dir = constants.UCF_101_FRAMES_DIR
        self.train_lines = open(train_split).readlines()
        self.test_lines = open(test_split).readlines()
        self.resnet_dims = constants.IMAGE_DIMS
        self.class_dict = get_ucf_101_dict()
        self.generator = ImageDataGenerator(zoom_range=0.4, horizontal_flip=0.3, rotation_range=15,
                                            shear_range=0.4)
        self.generate_longer = generate_longer
        self.augment_data = augment_data

    def parse_train_split_row(self, train_row):
        label = train_row.split(" ")[1]
        name_only = train_row.split(" ")[0]
        # class_name, file_name = name_only.split("/")[0], name_only.split("/")[1].split(".")[0]
        class_name, file_name = name_only.split("/")[0], name_only.split("/")[1]  # version for video
        return class_name, file_name, label

    def parse_test_split_row(self, test_row):
        # class_name, file_name = test_row.split("/")[0], test_row.split("/")[1].split(".")[0]
        class_name, file_name = test_row.split("/")[0], test_row.split("/")[1].strip()  # version for video
        return class_name, file_name

    def retrieve_train_data_gen(self):
        """
        Creates generator for train data
        :return: Name of file, list of frames, label
        """
        for row in self.train_lines:
            class_name, file_name, label = self.parse_train_split_row(row)
            final_name = class_name + "_" + file_name
            yield final_name, self.load_frames_from_video(class_name, file_name), label

    def retrieve_test_data_gen(self, parse_train=False):
        """
        Creates generator for test data
        :return: Name of file, list of frames, label
        """
        for row in self.test_lines:
            if parse_train:
                class_name, file_name, label = self.parse_train_split_row(row)
            else:
                class_name, file_name = self.parse_test_split_row(row)
                label = self.class_dict[class_name]
            final_name = class_name + "_" + file_name
            yield final_name, self.load_frames_from_video(class_name, file_name), label

    def get_length_of_video(self, capture):
        count = 0
        while (True):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if not ret:
                break

            count += 1
        return count

    def augment_image(self, img, inter=cv2.INTER_AREA):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (h, w) = img.shape[:2]

        # check to see if the width is None
        target_dim = constants.IMAGE_DIMS[0]
        target_height = target_dim + 30
        ratio = target_height / h
        new_width = int(ratio * w)
        dim = (new_width, target_height)

        # resize the image
        resized = cv2.resize(img, dim, interpolation=inter)

        if not self.augment_data:
            # MAKE CENTER CROP
            width_start = new_width // 2 - target_dim // 2
            width_end = width_start + target_dim

            height_start = target_height // 2 - target_dim // 2
            height_end = height_start + target_dim

            assert (width_end - width_start) == target_dim
            assert (height_end - height_start) == target_dim
            cropped_img = resized[height_start: height_end, width_start: width_end]

            assert cropped_img.shape[:2] == constants.IMAGE_DIMS
            return cropped_img

        random_start = np.random.randint(0, new_width - target_dim - 1)
        random_end = random_start + target_dim
        start_height = np.random.randint(0, target_height - target_dim - 1)
        end_height = start_height + target_dim
        assert (random_end - random_start) == target_dim
        assert (end_height - start_height) == target_dim
        cropped_img = resized[start_height: end_height, random_start: random_end]
        assert cropped_img.shape[:2] == constants.IMAGE_DIMS
        # return the resized image

        cropped_img = self.generator.random_transform(cropped_img)

        return cropped_img

    def load_frames_from_video(self, class_name, file_name):
        full_path = os.path.join(constants.UCF_101_DATA_DIR, class_name, file_name)
        cap = cv2.VideoCapture(full_path)
        length = self.get_length_of_video(capture=cap)
        cap = cv2.VideoCapture(full_path)
        frames = []
        success, image = cap.read()
        count = 0
        success = True

        if not self.generate_longer:
            target_length = constants.LSTM_SEQUENCE_LENGTH
        else:
            target_length = constants.LSTM_SEQUENCE_LENGTH_GENERATION

        indices_for_sequence = [int(a) for a in np.arange(0, length, length / target_length)]
        while success:
            if count in indices_for_sequence:
                image = self.augment_image(image)
                frames.append(image)
            success, image = cap.read()
            count += 1

        cap = cv2.VideoCapture(full_path)
        while len(frames) < target_length:
            success, image = cap.read()
            image = self.augment_image(image)
            frames.append(image)
        frames = frames[:target_length]
        return frames

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass