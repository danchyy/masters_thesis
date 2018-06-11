from base.base_model import BaseModel
from keras import applications
from keras.layers import Dense, Dropout, MaxPooling3D, Input, TimeDistributed, GlobalAveragePooling3D, AveragePooling1D, \
    Reshape, LSTM
from utils.util_script import get_number_of_classes
from keras.models import Model
from utils import constants
from keras import optimizers


class TimeDistributedCNN(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input_layer = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))
        # x = LSTM(1024, return_sequences=False, dropout=0.5)(input)
        # x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(input_layer)
        x = Dropout(0.5)(x)

        x = Dense(101, activation="softmax")(x)

        predictions = AveragePooling1D(pool_size=30)(x)
        self.model = Model(inputs=input_layer, outputs=predictions)
        print(self.model.output_shape)
        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
