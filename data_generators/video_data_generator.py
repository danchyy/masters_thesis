from keras import Input
from keras.engine import Model
from keras.layers import TimeDistributed
from keras.utils import Sequence
import os
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from utils import constants
from keras.applications.inception_v3 import preprocess_input, InceptionV3
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


class VideoDataGenerator(Sequence):
    """
    Class which generates batches from video input, in a way that it runs frames through pretrained inception net.
    That model is a time distributed model, in a sense that it takes sequences (30 frames) and outputs thirty frames.
    """

    def __init__(self, batch_size, split, num_of_classes):
        self.batch_size = batch_size
        self.split = split
        self.file_names = open(self.split).readlines()
        shuffle(self.file_names)
        self.num_of_classes = num_of_classes
        self.generator = ImageDataGenerator(zoom_range=0.2, horizontal_flip=0.2, rotation_range=20,
                                            shear_range=0.2, width_shift_range=0.5, height_shift_range=0.5,
                                            preprocessing_function=preprocess_input)
        input_layer = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.IMAGE_DIMS[0], constants.IMAGE_DIMS[1], 3),
                            name="input")
        pretrained_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

        for layer in pretrained_model.layers:
            layer.trainable = False

        x = TimeDistributed(pretrained_model)(input_layer)

        self.model = Model(inputs=input_layer, outputs=x)

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
        img = image.img_to_array(img)
        (h, w) = img.shape[:2]

        target_dim = constants.IMAGE_DIMS[0]
        # check to see if the width is None
        target_height = target_dim + 30
        ratio = target_height / h
        new_width = int(ratio * w)
        dim = (new_width, target_height)

        # resize the image
        resized = cv2.resize(img, dim, interpolation=inter)

        random_start = np.random.randint(0, new_width - target_dim - 1)
        random_end = random_start + target_dim
        start_height = np.random.randint(0, target_height - target_dim - 1)
        end_height = start_height + target_dim
        assert (random_end - random_start) == target_dim
        assert (end_height - start_height) == target_dim
        cropped_img = resized[start_height: end_height, random_start: random_end]
        assert cropped_img.shape[:2] == constants.IMAGE_DIMS

        cropped_img = self.generator.random_transform(cropped_img)

        return cropped_img

    def extract_frames_from_video(self, full_path):
        cap = cv2.VideoCapture(full_path)
        length = self.get_length_of_video(capture=cap)
        cap = cv2.VideoCapture(full_path)
        frames = []
        success, img = cap.read()
        count = 0
        success = True

        indices_for_sequence = [int(a) for a in np.arange(0, length, length / constants.LSTM_SEQUENCE_LENGTH)]
        indices_for_sequence = indices_for_sequence[:constants.LSTM_SEQUENCE_LENGTH]
        while success:
            if count in indices_for_sequence:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.augment_image(img)
                frames.append(img)
            success, img = cap.read()
            count += 1

        cap = cv2.VideoCapture(full_path)
        while len(frames) < constants.LSTM_SEQUENCE_LENGTH:
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.augment_image(img)
            frames.append(img)

        frames = frames[:constants.LSTM_SEQUENCE_LENGTH]
        frames = np.array(frames)
        return frames

    def __getitem__(self, index):
        start_index, end_index = index * self.batch_size, (index + 1) * self.batch_size
        curr_file_names = self.file_names[start_index: end_index]
        features, labels = [], []
        video_dir = constants.UCF_101_DATA_DIR
        for file_name in curr_file_names:
            curr_video, label = file_name.split(" ")[0].strip(), int(file_name.split(" ")[1])
            full_video_path = os.path.join(video_dir, curr_video)
            curr_frames = self.extract_frames_from_video(full_video_path)
            curr_frames = np.expand_dims(curr_frames, axis=0)
            curr_features = self.model.predict(curr_frames)
            updated_label = int(label) - 1  # SO WE CAN MOVE FIRST LABEL TO ZERO
            labels.append(updated_label)
            features.append(curr_features)
        features = np.array(features)
        return features, np.array(to_categorical(labels, num_classes=self.num_of_classes))

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        shuffle(self.file_names)

