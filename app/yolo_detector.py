from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        return self.model(frame, stream=True)
