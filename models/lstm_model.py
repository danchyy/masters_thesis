from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import optimizers, Input
from utils import constants
from utils.util_script import get_number_of_classes


class LSTMModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):

        input = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))
        x = LSTM(1024, return_sequences=False)(input)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=input, outputs=predictions)

        optimizer = optimizers.get(self.config.model.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.learning_rate
        if self.config.model.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=self.config.model.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
