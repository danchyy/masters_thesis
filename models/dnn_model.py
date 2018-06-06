from base.base_model import BaseModel
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, AveragePooling1D
from keras import optimizers, Input
from utils import constants
from keras.models import Model


class DNNModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))
        if self.config.model.architecture.available:
            first = True
            for layer_num in self.config.model.architecture.dense:
                if first:
                    x = TimeDistributed(Dense(layer_num, activation="relu"))(input)
                    first = False
                else:
                    x = TimeDistributed(Dense(layer_num, activation="relu"))(x)
                x = TimeDistributed(Dropout(0.5))(x)
        else:
            x = TimeDistributed(Dense(1024, activation="relu"))(input)
            x = TimeDistributed(Dropout(0.5))(x)
            x = TimeDistributed(Dense(1024, activation="relu"))(x)
            x = TimeDistributed(Dropout(0.5))(x)
        x = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        predictions = AveragePooling1D(pool_size=constants.LSTM_SEQUENCE_LENGTH)(x)
        predictions = Flatten()(predictions)
        self.model = Model(inputs=input, outputs=predictions)
        print(self.model.output.shape)
        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
