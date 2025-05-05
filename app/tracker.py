from  util.sort import  Sort


class Tracker:
    def __init__(self):
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    def update(self, detections):
        return self.tracker.update(detections)
