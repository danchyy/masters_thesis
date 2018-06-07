from utils.config import process_config
from utils.dirs import create_dirs
from utils import constants
from base.base_trainer import BaseTrain
import tensorflow as tf
from shutil import copyfile
import os
import inspect
from keras.backend.tensorflow_backend import set_session
import argparse
from utils import factory
import sys


config_dict = {
    "lstm": constants.LSTM_CONFIG,
    "fine_tuned": constants.FINE_TUNED_CONFIG,
    "dnn": constants.DNN_CONFIG,
    "time_cnn": constants.TIME_DISTRIBUTED_CNN_CONFIG,
    "avg_seq_cls": constants.AVERAGED_SEQUENCES_CLASSIFIER_CONFIG
}


def get_config_for_model(model_type):
    return config_dict[model_type] if model_type in config_dict else ""


def main(memory_frac, config, model_type):

    if memory_frac is None:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        set_session(tf.Session(config=tf_config))
    else:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = memory_frac
        set_session(tf.Session(config=tf_config))
    if model_type is None and config is None:
        print("Pass either --model_type argument or --config argument")
        sys.exit(1)
    if model_type is not None:
        config_path = get_config_for_model(model_type)
    else:
        config_path = config

    # process the json configuration file
    config = process_config(config_path)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.model_dir,
                 config.callbacks.config_dir, config.callbacks.result_dir])

    copyfile(config_path, os.path.join(config.callbacks.config_dir, "config.json"))

    print('Create the data generator.')
    data_loader = factory.create("data_loader." + config.data_loader.name)(config)

    print('Create the model.')
    model = factory.create("models." + config.model.name)(config)
    copyfile(inspect.getfile(model.__class__), os.path.join(config.callbacks.model_dir, "model.py"))

    print('Create the trainer')
    trainer = BaseTrain(model.build_model(), data_loader, config)

    print('Start training the model.')
    trainer.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training of deep neural net. '
                                                 'You need to pass either config or model_type.')
    parser.add_argument('--memory_frac', dest='memory_frac', type=float,
                        help='Fraction of gpu which will be used for training.\nIf nothing is passed, allow growth '
                             'option will be turned on.', required=False)
    parser.add_argument('--config', dest='config', type=str, help="Path to config which will be used for training.")
    parser.add_argument('--model_type', dest='model_type', type=str, help="Overrides the config argument and loads "
                                                                          "models with set up parameters:\n"
                                                                          "lstm - LSTMModel\n"
                                                                          "fine_tune - FineTuned\n"
                                                                          "dnn - DNNModel\n"
                                                                          "time_cnn - TimeDistributedCNN\n"
                                                                          "avg_seq_cls - AveragedSequencesClassifier")

    args = parser.parse_args()
    main(args.memory_frac, args.config, args.model_type)
