import os

import cv2
import glob

video_list = sorted(glob.glob('../video_sequence/*'))

for i, path in enumerate(video_list):
    ind = 0
    video = cv2.VideoCapture(path)
    os.makedirs(f'./real_video/seq{i}')
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imwrite(f'./real_video/seq{i}/frame{ind}.png', frame)
            ind += 1
        else:
            break
