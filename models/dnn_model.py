from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from utils import constants
from utils.util_script import get_number_of_classes


class DNNModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE)))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(384, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.config.exp.num_of_classes, activation="softmax"))
        optimizer = optimizers.get(self.config.model.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.learning_rate
        if self.config.model.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=self.config.model.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
