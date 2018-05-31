from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from utils.util_script import get_number_of_items


class FineTunedTrainer(BaseTrain):

    def __init__(self, model, data, config, validation_data):
        super().__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
        self.validation_data = validation_data
        self.train_split = self.config.trainer.train_split
        self.test_split = self.config.trainer.test_split

    def train(self):
        steps_per_epoch = get_number_of_items(self.train_split) // self.config.trainer.batch_size
        validation_steps = get_number_of_items(self.test_split) // self.config.trainer.batch_size
        history = self.model.fit_generator(
            generator=self.data,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.validation_data,
            validation_steps=validation_steps,
            callbacks=self.callbacks
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
