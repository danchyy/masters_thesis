import os
import constants


def create_folders():
    source_dir = constants.UCF_101_DATA_DIR
    dest_dir = constants.UCF_101_FRAMES_DIR

    class_names = os.listdir(source_dir)

    for class_name in class_names:
        if class_name.startswith("."):
            continue
        os.makedirs(os.path.join(dest_dir, class_name))
        video_names = os.listdir(os.path.join(source_dir, class_name))
        for video_name in video_names:
            name_only = video_name.split(".")[0]
            os.makedirs(os.path.join(dest_dir, class_name, name_only))

def create_for_feature_vectors():
    source_dir = constants.UCF_101_DATA_DIR
    dest_dir = constants.UCF_101_FEATURE_VECTORS

    class_names = os.listdir(source_dir)

    for class_name in class_names:
        if class_name.startswith("."):
            continue
        os.makedirs(os.path.join(dest_dir, class_name))

create_for_feature_vectors()