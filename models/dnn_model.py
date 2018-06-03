from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers, Input
from utils import constants
from keras.models import Model


class DNNModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))
        x = Flatten()(input)
        if self.config.model.architecture.available:
            for layer_num in self.config.model.architecture.dense:
                x = Dense(layer_num, activation="relu")(x)
                x = Dropout(0.5)(x)
        else:
            x = Dense(1024, activation="relu")(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation="relu")(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation="relu")(x)
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
