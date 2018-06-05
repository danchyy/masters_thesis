from base.base_model import BaseModel
from keras import applications
from keras.layers import Dense, Dropout, MaxPooling3D, Input, TimeDistributed, GlobalAveragePooling3D
from utils.util_script import get_number_of_classes
from keras.models import Model
from utils import constants
from keras import optimizers


class TimeDistributedCNN(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        input_layer = Input(shape=(constants.LSTM_SEQUENCE_LENGTH, constants.IMAGE_DIMS[0], constants.IMAGE_DIMS[1], 3),
                            name="input")
        pretrained_model = applications.InceptionV3(include_top=False, weights="imagenet")

        for layer in pretrained_model.layers:
            layer.trainable = False
        x = TimeDistributed(pretrained_model)(input_layer)
        x = MaxPooling3D(pool_size=(3, 2, 2), strides=(3, 2, 2))(x)
        x = GlobalAveragePooling3D()(x)
        if self.config.model.architecture.available:
            for layer_num in self.config.model.architecture.dense:
                x = Dense(layer_num, activation="relu")(x)
                x = Dropout(0.5)(x)
        else:
            x = Dense(1024, activation="relu")(x)
            x = Dropout(0.5)(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=input_layer, outputs=predictions)

        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer in ["adam", "rmsprop"]:
            optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
