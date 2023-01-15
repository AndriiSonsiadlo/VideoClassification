import cv2
import numpy as np

from solution1.objects.Singleton import Singleton


class VideoLoader(metaclass=Singleton):

    def load_video(self, path: str, img_resize, max_video_frames=0):
        capture = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                frame = cv2.resize(frame, img_resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_video_frames:
                    break
        finally:
            capture.release()
        return np.array(frames)

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]
