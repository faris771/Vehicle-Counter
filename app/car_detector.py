import cv2
import cvzone
from util.sort import *
import constants as const
from video_reader import VideoReader
from yolo_detector import YOLODetector
from tracker import Tracker
from car_counter import CarCounter


class CarDetectionApp:
    def __init__(self):
        self.reader = VideoReader(const.CARS_VIDEO_PATH)
        self.detector = YOLODetector(const.YOLO_PATH)
        self.tracker = Tracker()
        self.counter = CarCounter(const.LIMIT_LINE)
        self.area_mask = cv2.imread(const.CARS_AREA_MASK_PATH)

        fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi file encoding
        self.out = cv2.VideoWriter(const.OUTPUT_PATH, fourcc, 30,
                                   (const.VIDEO_WIDTH_PIXELS, const.VIDEO_HEIGHT_PIXELS))

    def draw_limit_line(self, frame):
        cv2.line(frame, (const.LIMIT_LINE['x1'], const.LIMIT_LINE['y1']),
                 (const.LIMIT_LINE['x2'], const.LIMIT_LINE['y2']), (0, 0, 255), 5)

    def draw_box_and_info(self, frame, x1, y1, x2, y2, id):
        box_width, box_height = x2 - x1, y2 - y1
        cx, cy = x1 + box_width // 2, y1 + box_height // 2

        cvzone.cornerRect(frame, (x1, y1, box_width, box_height,), colorR=(255, 0, 255), rt=2, l=9)
        cvzone.putTextRect(frame, f"{id}", (max(const.MIN_TEXT_X, x1), max(const.MIN_TEXT_Y, y1)), 1, 1,
                           colorR=(0, 0, 255))
        cvzone.putTextRect(frame, f"Car Count: {self.counter.get_count()}", (20, 30), 2, 1)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        self.counter.check_and_count(id, cx, cy)

    def run(self):
        while True:
            success, frame = self.reader.read_frame()
            if not success:
                break

            shaded_frame = cv2.bitwise_and(frame,
                                           self.area_mask)  # to only count cars in a certain shaded area shown as white space in the mask, and ignore  other areas
            yolo_results = self.detector.detect(shaded_frame)

            detections = np.empty(
                (0, 5))  # 5 columns, currently 0 rows, to stack them later, and to be sent to the tracker
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cls_name = const.YOLO_CLASSES[int(box.cls[0])]

                    if cls_name in const.TO_BE_DETECTED_VEHICLES and confidence > const.CONFIDENCE_THRESHOLD:
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2, confidence])))

            tracker_results = self.tracker.update(detections)
            self.draw_limit_line(frame)

            for result in tracker_results:
                x1, y1, x2, y2, id = map(int, result)
                self.draw_box_and_info(frame, x1, y1, x2, y2, id)

            self.out.write(frame)  # save fram to output video
            cv2.imshow(const.FRAME_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.reader.release()
        cv2.destroyAllWindows()
