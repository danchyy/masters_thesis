from keras import metrics
from sklearn.metrics import f1_score


def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy, f1_metric]

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError