from base.base_model import BaseModel
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from utils.util_script import get_number_of_classes
from keras.models import Model
from utils import constants
from keras import optimizers, losses


class FineTunedCNN(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        pretrained_model = applications.InceptionV3(include_top=False, weights="imagenet",
                                                    input_shape=(constants.IMAGE_DIMS[0], constants.IMAGE_DIMS[1], 3))

        for layer in pretrained_model.layers:
            layer.trainable = False
        x = pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(get_number_of_classes(), activation="softmax")(x)
        self.model = Model(inputs=pretrained_model.input, outputs=predictions)
        optimizer = optimizers.get(self.config.model.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.learning_rate
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
