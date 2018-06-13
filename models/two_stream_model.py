from base.base_model import BaseModel
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, AveragePooling1D, Average
from keras import optimizers, Input
from utils import constants
from keras.models import Model


class TwoStreamModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input_normal = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))
        input_flow = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.LSTM_FEATURE_SIZE))

        if self.config.model.architecture.available:
            for i in range(len(self.config.model.architecture.dense_normal)):
                dense_num = self.config.model.architecture.dense_normal[i]
                dropout_rate = self.config.model.architecture.dropout_normal[i]
                if i == 0:
                    x_normal = TimeDistributed(Dense(dense_num, activation="relu"))(input_normal)
                else:
                    x_normal = TimeDistributed(Dense(dense_num, activation="relu"))(x_normal)
                    x_normal = TimeDistributed(Dropout(dropout_rate))(x_normal)
            for i in range(len(self.config.model.architecture.dense_flow)):
                dense_num = self.config.model.architecture.dense_flow[i]
                dropout_rate = self.config.model.architecture.dropout_flow[i]
                if i == 0:
                    x_flow = TimeDistributed(Dense(dense_num, activation="relu"))(input_flow)
                else:
                    x_flow = TimeDistributed(Dense(dense_num, activation="relu"))(x_flow)
                    x_flow = TimeDistributed(Dropout(dropout_rate))(x_flow)
        else:
            x_normal = TimeDistributed(Dense(1024, activation="relu"))(input_normal)
            x_normal = TimeDistributed(Dropout(0.5))(x_normal)
            x_flow = TimeDistributed(Dense(1024, activation="relu"))(input_flow)
            x_flow = TimeDistributed(Dropout(0.5))(x_flow)

        # classifier
        x_normal = Dense(self.config.exp.num_of_classes, activation="softmax")(x_normal)
        predictions_normal = AveragePooling1D(pool_size=constants.LSTM_SEQUENCE_LENGTH)(x_normal)
        predictions_normal = Flatten()(predictions_normal)

        x_flow = Dense(self.config.exp.num_of_classes, activation="softmax")(x_flow)
        predictions_flow = AveragePooling1D(pool_size=constants.LSTM_SEQUENCE_LENGTH)(x_flow)
        predictions_flow = Flatten()(predictions_flow)

        averaged_predictions = Average([predictions_normal, predictions_flow])
        self.model = Model(inputs=[input_normal, input_flow], outputs=averaged_predictions)
        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer == "sgd":
            optimizer.nesterov = self.config.model.optimizing.nesterov
            optimizer.momentum = self.config.model.optimizing.momentum
        optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=self.metrics)
        return self.model