#!/bin/bash
python3 utils/prepare_data_for_cnn.py train_val01.txt test01.txt
python3 utils/prepare_data_for_cnn.py train_val02.txt test02.txt
python3 utils/prepare_data_for_cnn.py train_val02.txt test03.txt
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_1_1.json
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_1_2.json
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_1_3.json
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_2_1.json
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_2_2.json
python3 main.py --memory_frac 0.7 --config custom_configs/testing/fine_tune_configs/config_2_3.json