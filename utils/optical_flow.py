import cv2 as cv2
import numpy as np
from utils import constants
import os

video_path = "ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c02.avi"

cap = cv2.VideoCapture(os.path.join(constants.UCF_101_DATA_DIR, video_path))
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
count = 0
while(ret):
    ret, frame2 = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)
    k = cv2.waitKey(30)
    # cv2.imwrite('opticalfb.png', frame2)
    name = "flows/optical_" + str(count) + ".png"
    cv2.imwrite(name, bgr)
    prvs = next
cap.release()
cv2.destroyAllWindows()