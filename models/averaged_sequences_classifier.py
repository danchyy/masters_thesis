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

        if self.config.model.architecture.available:
            for i in range(len(self.config.model.architecture.dense)):
                dense_num = self.config.model.architecture.dense[i]
                dropout_rate = self.config.model.architecture.dropout[i]
                if i == 0:
                    x = Dense(dense_num, activation="relu")(input)
                else:
                    x = Dense(dense_num, activation="relu")(x)
                x = Dropout(dropout_rate)(x)
        else:
            x = Dense(1024, activation="relu")(input_layer)
            x = Dropout(0.5)(x)

        x = Flatten()(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=input_layer, outputs=predictions)
        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer == "sgd":
            optimizer.nesterov = self.config.model.optimizing.nesterov
            optimizer.momentum = self.config.model.optimizing.momentum
        optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=self.metrics)
        return self.model
