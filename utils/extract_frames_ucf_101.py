import cv2
import os
import constants


def extract_video_frames_for_video(video_name):
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()
    count = 0
    success = True
    file_name = os.path.basename(video_name)
    name_only = file_name.split(".")[0]
    class_name = os.path.basename(os.path.abspath(os.path.join(video_name, os.path.pardir)))
    while success:
        frame_name = "frame_" + str(count) + ".jpg"
        full_frame_path = os.path.join(constants.UCF_101_FRAMES_DIR, class_name, name_only, frame_name)
        res = cv2.imwrite(full_frame_path, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

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
            if count % 5 == 0:
                print("Progress for " + class_name + ": %d / %d" % (count, total))

        print("Ended with " + class_name)

if __name__ == '__main__':
    iterate_over_folders()