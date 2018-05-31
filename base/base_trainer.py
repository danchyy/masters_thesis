import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.callbacks = []

    def train(self):
        raise NotImplementedError

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
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

        if self.config.callbacks.early_stopping:
            self.callbacks.append(
                EarlyStopping(monitor="val_loss", mode="auto", patience=6)
            )
