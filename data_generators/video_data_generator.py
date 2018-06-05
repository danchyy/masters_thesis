from keras.utils import Sequence
import os
import json
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from utils import constants
import cv2


class VideoDataGenerator(Sequence):

    def __init__(self, batch_size, split, num_of_classes):
        self.batch_size = batch_size
        self.split = split
        self.file_names = open(self.split).readlines()
        shuffle(self.file_names)
        self.num_of_classes = num_of_classes

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
        (h, w) = img.shape[:2]

        # check to see if the width is None
        target_height = constants.IMAGE_DIMS[0]
        ratio = target_height / h
        new_width = int(ratio * w)
        dim = (new_width, target_height)

        # resize the image
        resized = cv2.resize(img, dim, interpolation=inter)

        random_start = np.random.randint(0, new_width - target_height - 1)
        random_end = random_start + target_height
        assert (random_end - random_start) == target_height
        cropped_img = resized[:, random_start: random_end]
        assert cropped_img.shape[:2] == constants.IMAGE_DIMS
        # return the resized image

        if np.random.rand() < 0.3:
            cropped_img = cv2.flip(cropped_img, 0)  # horizontal flip
        if np.random.rand() < 0.3:
            cropped_img = cv2.flip(cropped_img, 1)  # vertical flip

        return cropped_img

    def extract_frames_from_video(self, full_path):
        cap = cv2.VideoCapture(full_path)
        length = self.get_length_of_video(capture=cap)
        cap = cv2.VideoCapture(full_path)
        frames = []
        success, image = cap.read()
        count = 0
        success = True

        indices_for_sequence = [int(a) for a in np.arange(0, length, length / constants.LSTM_SEQUENCE_LENGTH)]
        indices_for_sequence = indices_for_sequence[:constants.LSTM_SEQUENCE_LENGTH]
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if count in indices_for_sequence:
                image = self.augment_image(image)
                frames.append(image)
            success, image = cap.read()
            count += 1

        cap = cv2.VideoCapture(full_path)
        while len(frames) < constants.LSTM_SEQUENCE_LENGTH:
            success, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.augment_image(image)
            frames.append(image)
        frames = frames[:constants.LSTM_SEQUENCE_LENGTH]
        return frames

    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        frames, labels = [], []
        video_dir = constants.UCF_101_DATA_DIR
        for file_name in curr_file_names:
            curr_video, label = file_name.split(" ")[0].strip(), int(file_name.split(" ")[1])
            full_video_path = os.path.join(video_dir, curr_video)
            curr_frame = self.extract_frames_from_video(full_video_path)
            updated_label = int(label) - 1  # SO WE CAN MOVE FIRST LABEL TO ZERO
            labels.append(updated_label)
            frames.append(curr_frame)
        return np.array(frames), np.array(to_categorical(labels, num_classes=self.num_of_classes))

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        shuffle(self.file_names)

