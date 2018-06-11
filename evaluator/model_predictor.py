from utils import constants, factory
import os
import numpy as np
from keras.models import Model
from utils.util_script import get_ucf_101_label_dict

class ModelPredictor:

    def __init__(self, config, weights_path, is_data_features=True):
        self.config = config
        self.weights_path = weights_path
        self.model = None
        self.build_model()
        print(f"Loading model checkpoint {self.weights_path} ...")
        self.model.load_weights(self.weights_path)
        print("Model loaded")
        if is_data_features:
            self.root = os.path.join(constants.UCF_101_LSTM_DATA, "test")
        self.label_dict = get_ucf_101_label_dict()

    def build_model(self):
        print('Create the model.')
        self.model = factory.create("models." + self.config.model.name)(self.config).build_model()

    def predict(self, video_path):
        """
        Returns top 10 predictions for video and loaded model
        :param video_path: Path to features from video
        :return:
        """
        features = np.load(video_path)
        features = features[:30]
        features = np.expand_dims(features, axis=0)
        assert isinstance(self.model, Model)
        softmax_output = self.model.predict(features)[0]
        top_10_labels = np.argsort(softmax_output)[::-1][:10]
        outputs = []
        for index, label in enumerate(top_10_labels):
            predicted_class = self.label_dict[label + 1]
            # print(f"{index+1}. {predicted_class} : {softmax_output[label]}")
            percentage_format = "{:.3f}".format(float(softmax_output[label] * 100))
            outputs.append(str(index + 1) + ". " + predicted_class + " : " + percentage_format + "%")
        return outputs

