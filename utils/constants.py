import os

ROOT_FOLDER = "/Users/daniel/myWork/masters_thesis"

UCF_101_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101")

UCF_101_CNN_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data")

UCF_101_FRAMES_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_frames")

UCF_101_DATA_SPLITS = os.path.join(ROOT_FOLDER, "data_splits")

UCF_101_CLASS_FILE_NAME = "classInd.txt"

UCF_101_NUMPY_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_numpy")

UCF_101_FEATURE_VECTORS = os.path.join(ROOT_FOLDER, "data", "UCF_101_feature_vectors")

IMAGE_DIMS = (299, 299)

UCF_101_SEQUENCE_FRAMES_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_sequence_frames")

UCF_101_LSTM_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM")

UCF_101_LSTM_DATA_train01 = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM_augmented")

FINE_TUNED_CONFIG = os.path.join(ROOT_FOLDER, "configs", "fine_tune_config.json")

LSTM_CONFIG = os.path.join(ROOT_FOLDER, "configs", "lstm_config.json")

DNN_CONFIG = os.path.join(ROOT_FOLDER, "configs", "dnn_config.json")

LSTM_SEQUENCE_LENGTH = 30

LSTM_FEATURE_SIZE = 2048
