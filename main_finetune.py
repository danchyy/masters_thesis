from data_loader.fine_tuned_data_loader import FineTunedDataLoader
from models.fine_tuned_cnn import FineTunedCNN
from trainers.fine_tuned_trainer import FineTunedTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import constants


def main():
    config_path = constants.FINE_TUNED_CONFIG
    # process the json configuration file
    config = process_config(config_path)


    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = FineTunedDataLoader(config)

    print('Create the model.')
    model = FineTunedCNN(config)

    print('Create the trainer')
    trainer = FineTunedTrainer(model.build_model(), data_loader.get_train_data(), config, data_loader.get_test_data())

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()