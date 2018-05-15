
import os
import constants
from collections import Counter
import matplotlib.pyplot as plt

classes = os.listdir(constants.UCF_101_FRAMES_DIR)

minimal, maximal = None, None
appearance_counter = Counter()
for class_name in classes:
    if class_name.startswith("."):
        continue

    video_files = os.listdir(os.path.join(constants.UCF_101_FRAMES_DIR, class_name))
    for video in video_files:
        if video.startswith("."):
            continue
        path = os.path.join(constants.UCF_101_FRAMES_DIR, class_name, video)
        number_of_frames = len(os.listdir(path))
        appearance_counter[number_of_frames] += 1

for item in sorted(appearance_counter.most_common()):
    plt.bar(item[0], item[1])
    print("Number of frames: %d, appearances: %d" % (item[0], item[1]))

plt.show()
