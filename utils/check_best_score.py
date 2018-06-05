from utils import constants
import os


def walkdir(dirname):
    for cur, _dirs, files in os.walk(dirname):
        for f in files:
            if f == "best_val_acc.txt":
                print(os.path.join(cur, f))
                print("VAL_ACC: " + open(os.path.join(cur, f)).read())


root = os.path.join(constants.ROOT_FOLDER, "experiments")
walkdir(root)

