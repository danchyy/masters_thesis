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
            layer.trainable = self.config.model.train_lower
        x = pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        if self.config.model.architecture.available:
            for i in range(len(self.config.model.architecture.dense)):
                dense_num = self.config.model.architecture.dense[i]
                dropout_rate = self.config.model.architecture.dropout[i]
                x = Dense(dense_num, activation="relu")(x)
                x = Dropout(dropout_rate)(x)
        else:
            x = Dense(2048, activation="relu")(x)
            x = Dropout(0.5)(x)
        predictions = Dense(self.config.exp.num_of_classes, activation="softmax")(x)
        self.model = Model(inputs=pretrained_model.input, outputs=predictions)

        optimizer = optimizers.get(self.config.model.optimizing.optimizer)
        assert isinstance(optimizer, optimizers.Optimizer)
        optimizer.lr = self.config.model.optimizing.learning_rate
        if self.config.model.optimizing.optimizer == "sgd":
            optimizer.nesterov = self.config.model.optimizing.nesterov
            optimizer.momentum = self.config.model.optimizing.momentum
        optimizer.decay = self.config.model.decay
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=self.metrics)
        return self.model
