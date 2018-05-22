import os

ROOT_FOLDER = "/Users/daniel/myWork/masters_thesis"

UCF_101_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101")

UCF_101_CNN_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data")

UCF_101_FRAMES_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_frames")

UCF_101_TRAIN_TEST_SPLIT_CLASS_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_TrainTestlist")

UCF_101_CLASS_FILE_NAME = "classInd.txt"

UCF_101_NUMPY_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_numpy")

UCF_101_FEATURE_VECTORS = os.path.join(ROOT_FOLDER, "data", "UCF_101_feature_vectors")

RESNET_DIMS = (224, 224)

UCF_101_LSTM_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM")

FINE_TUNED_CONFIG = os.path.join(ROOT_FOLDER, "configs", "fine_tune_config.json")
