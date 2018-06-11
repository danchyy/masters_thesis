from utils import constants
import os
import json
from copy import deepcopy

counter = 1


def change_data_and_save(json_data, index, optimizer=None, learning_rate=None, batch=None, dropout=None,
                         bidirectional=None, return_seq=None, dense_drop=None):
    global counter
    copied_json = deepcopy(json_data)
    if optimizer is not None:
        copied_json["model"]["optimizing"]["optimizer"] = optimizer
    if learning_rate is not None:
        copied_json["model"]["optimizing"]["learning_rate"] = learning_rate
    if batch is not None:
        copied_json["trainer"]["batch_size"] = batch
    if dropout is not None:
        copied_json["model"]["architecture"]["dropout"] = dropout
    if bidirectional is not None:
        copied_json["model"]["architecture"]["bidirectional"] = bidirectional
    if return_seq is not None:
        copied_json["model"]["return_sequence"] = return_seq
    if dense_drop is not None:
        copied_json["model"]["architecture"]["dense_dropout"] = dense_drop
    copied_json["callbacks"]["early_stopping"]["available"] = False
    if learning_rate == 0.0001:
        copied_json["trainer"]["num_epochs"] = 100
    else:
        copied_json["trainer"]["num_epochs"] = 200
    if optimizer == "sgd":
        copied_json["trainer"]["num_epochs"] = 200
        copied_json["model"]["optimizing"]["momentum"] = 0.9
        copied_json["model"]["optimizing"]["nesterov"] = True
    with open(os.path.join(target_dir, "config_" + index + "_" + str(counter) + ".json"), "w") as f:
        json.dump(copied_json, f, indent=3)
    counter += 1


target_config_dir = "avg_seq_cls_configs"
# 10 -> dnn
# 11 -> avg_seq
# 2 -> lstm
# 2 -> fine_tune
target_config = "config_11.json"
is_lstm = True if "lstm" in target_config_dir else False
index = target_config.split(".")[0].split("_")[1]

configs_dir = constants.CUSTOM_CONFIGS_DIR
source_config = os.path.join(configs_dir, target_config_dir, target_config)
target_dir = os.path.join(configs_dir, "hyperparameter_search", target_config_dir)

with open(source_config) as f:
    json_data = json.load(f)

batch_values = [32, 64, 128]
learning_rates = [0.0001, 0.00001]
optimizers = ["sgd"]
dropouts = [[0.5], [0.2]]
dense_dropouts = [[0.5], [0.2]]
returned_seq = False
bidirectional = [True, False]

for batch in batch_values:
    for optimizer in optimizers:
        for lr in learning_rates:
            change_data_and_save(json_data, index, optimizer=optimizer, learning_rate=lr, batch=batch)

"""## LSTM
for lr in learning_rates:
    # for drop in dropouts:
    #for dense_drop in dense_dropouts:
    for optimizer in optimizers:
            for bi in bidirectional:
                change_data_and_save(json_data, index, learning_rate=lr, optimizer=optimizer,
                                     bidirectional=bi)
"""


"""for optimizer in optimizers:
    change_data_and_save(json_data, index, optimizer=optimizer)

for lr in learning_rates:
    change_data_and_save(json_data, index, learning_rate=lr)

for batch in batch_values:
    change_data_and_save(json_data, index, batch=batch)

for dropout in dropouts:
    target_array = dropout * len(json_data["model"]["architecture"]["dropout"])
    change_data_and_save(json_data, index, dropout=target_array)

if is_lstm:
    change_data_and_save(json_data, index, return_seq=returned_seq)
    change_data_and_save(json_data, index, bidirectional=bidirectional)
    for dropout in dense_dropouts:
        target_array = dropout * len(json_data["model"]["architecture"]["dense_dropout"])
        change_data_and_save(json_data, index, dense_drop=target_array)"""
