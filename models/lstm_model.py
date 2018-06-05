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
        if self.config.model.architecture.available:
            for i in range(len(self.config.model.architecture.lstm)):
                neurons = self.config.model.architecture.lstm[i]
                dropout = self.config.model.architecture.dropout[i]
                return_sequences = i != (len(self.config.model.architecture.lstm)-1)
                if i == 0:
                    x = LSTM(neurons, return_sequences=return_sequences, dropout=dropout)(input)
                else:
                    x = LSTM(neurons, return_sequences=return_sequences, dropout=dropout)(x)
            for neurons in self.config.model.architecture.dense:
                x = Dense(neurons, activation="relu")(x)
        else:
            x = LSTM(1024, return_sequences=False, dropout=0.5)(input)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=input, outputs=predictions)

        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
