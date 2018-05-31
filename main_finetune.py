from data_loader.fine_tuned_data_loader import FineTunedDataLoader
from models.fine_tuned_cnn import FineTunedCNN
from trainers.fine_tuned_trainer import FineTunedTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import constants
import tensorflow as tf
from shutil import copyfile
import os
import inspect
from keras.backend.tensorflow_backend import set_session
import argparse


def main(memory_frac):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = memory_frac
    set_session(tf.Session(config=tf_config))

    config_path = constants.FINE_TUNED_CONFIG
    # process the json configuration file
    config = process_config(config_path)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.model_dir])

    print('Create the data generator.')
    data_loader = FineTunedDataLoader(config)

    print('Create the model.')
    model = FineTunedCNN(config)

    copyfile(inspect.getfile(model.__class__), os.path.join(config.callbacks.model_dir, "model.py"))

    print('Create the trainer')
    trainer = FineTunedTrainer(model.build_model(), data_loader.get_train_data(), config, data_loader.get_test_data())

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training of net')
    parser.add_argument('--memory_frac', dest='memory_frac', type=float,
                        help='Fractiong of gpu which will be used for training', required=True)

    args = parser.parse_args()
    main(args.memory_frac)
