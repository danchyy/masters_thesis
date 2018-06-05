import cv2
import os
import constants
import numpy as np

def get_length_of_video(capture):
    count = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = capture.read()
        if not ret:
            break

        count += 1
    return count


def extract_video_frames_for_video(video_name):
    vidcap = cv2.VideoCapture(video_name)
    length = get_length_of_video(capture=vidcap)
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()
    file_name = os.path.basename(video_name)
    name_only = file_name.split(".")[0]

    indices_for_sequence = [int(a) for a in np.arange(0, length, length / constants.LSTM_SEQUENCE_LENGTH)]
    class_name = os.path.basename(os.path.abspath(os.path.join(video_name, os.path.pardir)))
    added = []
    count = 0
    if not os.path.exists(os.path.join(constants.UCF_101_FRAMES_DIR, class_name)):
        os.makedirs(os.path.join(constants.UCF_101_FRAMES_DIR, class_name))
    if not os.path.exists(os.path.join(constants.UCF_101_FRAMES_DIR, class_name, name_only)):
        os.makedirs(os.path.join(constants.UCF_101_FRAMES_DIR, class_name, name_only))
    while success:
        if count in indices_for_sequence:
            frame_name = "frame_" + str(count) + ".jpg"
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            full_frame_path = os.path.join(constants.UCF_101_FRAMES_DIR, class_name, name_only, frame_name)
            res = cv2.imwrite(full_frame_path, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    cap = cv2.VideoCapture(video_name)
    while len(added) < constants.LSTM_SEQUENCE_LENGTH:
        success, image = cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_name = "frame_" + str(count) + ".jpg"
        added.append(frame_name)
        full_frame_path = os.path.join(constants.UCF_101_FRAMES_DIR, class_name, name_only, frame_name)
        res = cv2.imwrite(full_frame_path, image)  # save frame as JPEG file
        count += 1
    assert len(added) == constants.LSTM_SEQUENCE_LENGTH


def iterate_over_folders():
    class_names = os.listdir(constants.UCF_101_DATA_DIR)
    print("Class names:")
    print(class_names)
    for class_name in class_names:
        if class_name.startswith("."):
            continue
        print("Started with " + class_name)
        videos = os.listdir(os.path.join(constants.UCF_101_DATA_DIR, class_name))
        count = 0
        total = len(videos)
        for video in videos:
            full_video_path = os.path.join(constants.UCF_101_DATA_DIR, class_name, video)
            extract_video_frames_for_video(full_video_path)
            count += 1
            if count % 10 == 0:
                print("Progress for " + class_name + ": %d / %d" % (count, total))

        print("Ended with " + class_name)


if __name__ == '__main__':
    iterate_over_folders()