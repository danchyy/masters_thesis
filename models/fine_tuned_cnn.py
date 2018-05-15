from base.base_model import BaseModel
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from utils.util_script import get_number_of_classes
from keras import optimizers, losses

class FineTunedCNN(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        self.model = Sequential()
        pretrained_model = applications.ResNet50(include_top=False, weights="imagenet")
        self.model.add(pretrained_model)
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(get_number_of_classes(), activation='softmax'))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "precision",
                                                                                       "accuracy"])
        return self.model
