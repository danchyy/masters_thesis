import os

ROOT_FOLDER = "/Users/daniel/myWork/masters_thesis"

UCF_101_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101")

UCF_101_CNN_DATA_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data")

UCF_101_CNN_DATA_DIR_1 = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data_split1")

UCF_101_CNN_DATA_DIR_2 = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data_split2")

UCF_101_CNN_DATA_DIR_3 = os.path.join(ROOT_FOLDER, "data", "UCF_101_cnn_data_split3")

UCF_101_FRAMES_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_frames")

UCF_101_DATA_SPLITS = os.path.join(ROOT_FOLDER, "data_splits")

UCF_101_CLASS_FILE_NAME = "classInd.txt"

UCF_101_NUMPY_DIR = os.path.join(ROOT_FOLDER, "data", "UCF_101_numpy")

UCF_101_FEATURE_VECTORS = os.path.join(ROOT_FOLDER, "data", "UCF_101_feature_vectors")

IMAGE_DIMS = (299, 299)

UCF_101_SEQUENCE_FRAMES_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_sequence_frames")

UCF_101_LSTM_DATA = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM")

UCF_101_LSTM_DATA_AUGMENT = os.path.join(ROOT_FOLDER, "data", "UCF_101_LSTM_augmented")

UCF_101_EXTRACTED_FEATURES = os.path.join(ROOT_FOLDER, "data", "UCF_101_extracted_features")

UCF_101_EXTRACTED_FEATURES_NO_GLOB_POOL = os.path.join(ROOT_FOLDER, "data", "UCF_101_extracted_features_no_glob_pool")

UCF_101_EXTRACTED_FEATURES_TEST_1 = os.path.join(ROOT_FOLDER, "data", "UCF_101_extracted_features_test_1")

UCF_101_EXTRACTED_FLOW = os.path.join(ROOT_FOLDER, "data", "UCF_101_extracted_flow")

FINE_TUNED_CONFIG = os.path.join(ROOT_FOLDER, "configs", "fine_tune_config.json")

LSTM_CONFIG = os.path.join(ROOT_FOLDER, "configs", "lstm_config.json")

DNN_CONFIG = os.path.join(ROOT_FOLDER, "configs", "dnn_config.json")

CUSTOM_CONFIGS_DIR = os.path.join(ROOT_FOLDER, "custom_configs")

TIME_DISTRIBUTED_CNN_CONFIG = os.path.join(ROOT_FOLDER, "configs", "time_distributed_cnn_config.json")

AVERAGED_SEQUENCES_CLASSIFIER_CONFIG = os.path.join(ROOT_FOLDER, "configs", "averaged_sequences_classifier_config.json")

TWO_STREAM_CONFIG = os.path.join(ROOT_FOLDER, "configs", "two_stream_config.json")

EXPORTED_MODELS_DIR = os.path.join(ROOT_FOLDER, "exported_models")

LOCAL_VIDEO_DATA_FOLDER = "/Users/daniel/myWork/masters_thesis/data/UCF_101"

LOCAL_EXTRACTED_FEATURES_DATA = "/Users/daniel/myWork/masters_thesis/data/UCF_101_extracted_features"

LSTM_SEQUENCE_LENGTH = 30

LSTM_SEQUENCE_LENGTH_GENERATION = 50

LSTM_FEATURE_SIZE = 2048
