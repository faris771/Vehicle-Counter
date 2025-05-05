import constants as const
import cv2



class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(const.WIDTH_ID, const.VIDEO_WIDTH_PIXELS)
        self.cap.set(const.HEIGHT_ID, const.VIDEO_HEIGHT_PIXELS)

    def read_frame(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.resize(frame, (const.VIDEO_WIDTH_PIXELS, const.VIDEO_HEIGHT_PIXELS))
        return success, frame

    def release(self):
        self.cap.release()
