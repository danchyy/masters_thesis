import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils.util_script import get_number_of_items
from keras.models import Model


class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.callbacks = []
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

    def evaluate(self):
        assert isinstance(self.model, Model)
        scores = self.model.evaluate_generator(self.validation_generator, get_number_of_items(self.test_split))
        scores_dict = {}
        for i in range(len(scores)):
            metric_name, score = self.model.metrics_names[i], scores[i]
            print(metric_name + " : " + str(score))
            scores_dict[metric_name] = score
        return scores_dict


    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                period=self.config.callbacks.save_every_n
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            ))

        if self.config.callbacks.early_stopping.available:
            self.callbacks.append(
                EarlyStopping(monitor=self.config.callbacks.early_stopping.monitor, mode="auto",
                              patience=self.config.callbacks.early_stopping.patience)
            )
