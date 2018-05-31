from data_loader.lstm_data_loader import LSTMDataLoader
from models.lstm_model import LSTMModel
from trainers.lstm_trainer import LSTMTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import constants
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def main():

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    config_path = constants.LSTM_CONFIG
    # process the json configuration file
    config = process_config(config_path)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = LSTMDataLoader(config)

    print('Create the model.')
    model = LSTMModel(config)

    print('Create the trainer')
    trainer = LSTMTrainer(model.build_model(), data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
