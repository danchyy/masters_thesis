from utils.config import process_config
import tensorflow as tf
from utils import constants
from shutil import copyfile
import os
from keras.backend.tensorflow_backend import set_session
from evaluator.model_predictor import ModelPredictor
import argparse
from utils import factory


def main(memory_frac, config, target_file):

    if memory_frac is None:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        set_session(tf.Session(config=tf_config))
    else:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = memory_frac
        set_session(tf.Session(config=tf_config))

    config_path = config

    # process the json configuration file
    config = process_config(config_path)

    print('Create the model.')
    model = factory.create("models." + config.model.name)(config)

    # checkpoint_path = os.path.join(constants.ROOT_FOLDER, config.callbacks.checkpoint_dir, config.exp.name + ".hdf5")
    checkpoint_path = "/Users/daniel/myWork/masters_thesis/experiments/dnn_dense_1024_adam_0.001_32/" \
                      "2018-06-06-19-21-29/checkpoints/dnn.hdf5"
    print(checkpoint_path)
    predictor = ModelPredictor(config, checkpoint_path)

    predictor.predict(target_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training of deep neural net. '
                                                 'You need to pass either config or model_type.')
    parser.add_argument("--target_file", dest='target_file', type=str, help="Path to file which wants to be predicted",
                        required=True)
    parser.add_argument('--memory_frac', dest='memory_frac', type=float,
                        help='Fraction of gpu which will be used for training.\nIf nothing is passed, allow growth '
                             'option will be turned on.', required=False)
    parser.add_argument('--config', dest='config', type=str, help="Path to config which will be used for training.")

    args = parser.parse_args()
    main(args.memory_frac, args.config, args.target_file)
