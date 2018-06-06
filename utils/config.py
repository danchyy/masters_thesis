import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    start_name = config.exp.name
    name = start_name
    if config.model.architecture.available:
        name = name + "_bi" if config.model.architecture.bidirectional else name
        if "lstm" in config.model.architecture:
            name += "_lstm"
            for layer_dim in config.model.architecture.lstm:
                name += "_" + str(layer_dim)
        if "dense" in config.model.architecture:
            name += "_dense"
            for layer_dim in config.model.architecture.dense:
                name += "_" + str(layer_dim)
    name += "_" + config.model.optimizing.optimizer
    name += "_" + str(config.model.optimizing.learning_rate)
    name += "_" + str(config.trainer.batch_size)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", name, time.strftime("%Y-%m-%d-%H-%M-%S/",
                                                        time.localtime()), "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", name, time.strftime("%Y-%m-%d-%H-%M-%S/",
                                                        time.localtime()), "checkpoints/")
    config.callbacks.model_dir = os.path.join("experiments", name, time.strftime("%Y-%m-%d-%H-%M-%S/",
                                                        time.localtime()), "model/")
    config.callbacks.config_dir = os.path.join("experiments", name, time.strftime("%Y-%m-%d-%H-%M-%S/",
                                                        time.localtime()), "config/")
    config.callbacks.result_dir = os.path.join("experiments", name, time.strftime("%Y-%m-%d-%H-%M-%S/",
                                                        time.localtime()), "results/")
    return config
