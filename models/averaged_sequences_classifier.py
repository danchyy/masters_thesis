from base.base_model import BaseModel
from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model
from utils import constants
from keras import optimizers

class AveragedSequencesClassifier(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input_layer = Input(shape=(1, constants.LSTM_FEATURE_SIZE))
        x = Dense(1024, activation="relu")(input_layer)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=input_layer, outputs=predictions)
        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=self.metrics)
        return self.model
