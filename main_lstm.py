from data_loader.lstm_data_loader import LSTMDataLoader
from models.lstm_model import LSTMModel
from trainers.lstm_trainer import LSTMTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import constants
import tensorflow as tf
from shutil import copyfile
import os
import inspect
from keras.backend.tensorflow_backend import set_session
import argparse


def main(memory_frac, config):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = memory_frac
    set_session(tf.Session(config=tf_config))
    if config is None:
        config_path = constants.LSTM_CONFIG
    else:
        config_path = config
    # process the json configuration file
    config = process_config(config_path)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.model_dir,
                 config.callbacks.config_dir, config.callbacks.result_dir])

    print('Create the data generator.')
    data_loader = LSTMDataLoader(config)

    print('Create the model.')
    model = LSTMModel(config)

    copyfile(inspect.getfile(model.__class__), os.path.join(config.callbacks.model_dir, "model.py"))
    copyfile(config_path, os.path.join(config.callbacks.config_dir, "config.json"))

    print('Create the trainer')
    trainer = LSTMTrainer(model.build_model(), data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training of net')
    parser.add_argument('--memory_frac', dest='memory_frac', type=float,
                        help='Fraction of gpu which will be used for training', required=True)
    parser.add_argument('--config', dest='config', type=str, help="Path to config which will be used for training")

    args = parser.parse_args()
    main(args.memory_frac, args.config)
