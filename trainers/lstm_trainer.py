import os
from base.base_trainer import BaseTrain
from utils.util_script import get_number_of_items

class LSTMTrainer(BaseTrain):

    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
        self.train_generator = self.data.get_train_data()
        self.validation_generator = self.data.get_test_data()
        self.train_split = self.config.trainer.train_split
        self.test_split = self.config.trainer.test_split

    def train(self):
        steps_per_epoch = get_number_of_items(self.train_split) // self.config.trainer.batch_size
        validation_steps = get_number_of_items(self.test_split) // self.config.trainer.batch_size
        history = self.model.fit_generator(
            generator=self.train_generator,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=self.callbacks
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

        result_dir = self.config.callbacks.result_dir
        for metric in ['loss', 'acc', 'val_loss', 'val_acc']:
            values = [str(value) + "\n" for value in history.history[metric]]
            open(os.path.join(result_dir, metric + ".txt"), "w").writelines(values)
        open(os.path.join(result_dir, "best_val_acc.txt"), "w").write(str(max(self.val_acc)))
