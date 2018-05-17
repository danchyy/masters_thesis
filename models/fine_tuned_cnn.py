from base.base_model import BaseModel
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from utils.util_script import get_number_of_classes
from utils import constants
from keras import optimizers, losses

class FineTunedCNN(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        self.model = Sequential()
        pretrained_model = applications.ResNet50(include_top=False, weights="imagenet",
                                                 input_shape=(constants.RESNET_DIMS[0], constants.RESNET_DIMS[1], 3))
        for layer in pretrained_model.layers:
            layer.trainable = False
        self.model.add(pretrained_model)
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(get_number_of_classes(), activation='softmax'))
        optimizer = optimizers.get(self.config.model.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.learning_rate
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model
