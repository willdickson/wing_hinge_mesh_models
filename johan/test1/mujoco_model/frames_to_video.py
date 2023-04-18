import os
import cv2
import numpy as np

frames_dir = 'frames'
frame_list = os.listdir(frames_dir)
frame_list = [f for f in frame_list if os.path.splitext(f)[1]=='.npy']
frame_list = [os.path.join(os.curdir, frames_dir, f) for f in frame_list]
frame_list.sort()

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')


for i, name in enumerate(frame_list[::2]):

    print(name)
    frame = np.load(name)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if i==0:
        w = frame.shape[1]
        h = frame.shape[0]
        vid_out = cv2.VideoWriter('output.avi', fourcc, 30.0, (w, h))

    vid_out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break





