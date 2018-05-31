import os

ROOT_FOLDER = "/Users/daniel/myWork/masters_thesis"

UCF_101_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101")

UCF_101_CNN_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data")

UCF_101_CNN_DATA_DIR_TRAINLIST01 = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data_trainlist01")

UCF_101_FRAMES_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_frames")

UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_TrainTestlist")

UCF_101_CLASS_FILE_NAME = "classInd.txt"

UCF_101_NUMPY_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_numpy")

UCF_101_FEATURE_VECTORS = os.path.join(ROOT_FOLDER, "data", "UCF_101_feature_vectors")

IMAGE_DIMS = (299, 299)

UCF_101_LSTM_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM")

FINE_TUNED_CONFIG = os.path.join(ROOT_FOLDER, "configs", "fine_tune_config.json")

LSTM_CONFIG = os.path.join(ROOT_FOLDER, "configs", "lstm_config.json")

LSTM_SEQUENCE_LENGTH = 40

LSTM_FEATURE_SIZE = 2048
