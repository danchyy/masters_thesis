from utils.config import process_config
import tensorflow as tf
from base.base_trainer import BaseTrain
import os
from keras.backend.tensorflow_backend import set_session
import argparse
from utils import factory


def main(model_dir):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    config = os.path.join(model_dir, "config.json")
    config_path = config

    # process the json configuration file
    config = process_config(config_path)
    config.trainer.batch_size = 1

    print('Create the data generator.')
    data_loader = factory.create("data_loader." + config.data_loader.name)(config)

    print('Create the model.')
    model = factory.create("models." + config.model.name)(config)

    print('Create the trainer')
    trainer = BaseTrain(model.build_model(), data_loader, config)

    # checkpoint_path = os.path.join(constants.ROOT_FOLDER, config.callbacks.checkpoint_dir, config.exp.name + ".hdf5")
    checkpoint_path = os.path.join(model_dir, "model.hdf5")
    model.load(checkpoint_path)
    scores_dict = trainer.evaluate()
    for metric in scores_dict:
        open(os.path.join(model_dir, metric + ".txt"), "w").write(str(scores_dict[metric]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for testing of deep neural net. directory of model must be '
                                                 'passed')
    parser.add_argument('--model_dir', dest='model_dir', type=str, help="Path to model directory"
                                                                        " which will be used for evaluation.")

    args = parser.parse_args()
    main(args.model_dir)
