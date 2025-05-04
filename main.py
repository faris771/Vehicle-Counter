from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import constants as const
from util.sort import *


def main():

    cap = cv2.VideoCapture(const.CARS_VIDEO_PATH)
    cap.set(const.WIDTH_ID, const.VIDEO_WIDTH_PIXELS)
    cap.set(const.HEIGHT_ID, const.VIDEO_HEIGHT_PIXELS)

    yolo_model = YOLO(const.YOLO_PATH)
    area_mask = cv2.imread(const.CARS_AREA_MASK_PATH)

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # max_age=20: This parameter specifies the maximum number of consecutive frames an object can be missing before it is removed from the tracker.
    # min_hits=3: This defines the minimum number of consecutive frames an object must be detected before it is considered a valid track.
    # iou_threshold=0.3: This is the Intersection over Union (IoU) threshold used to associate detections with existing tracks. If the IoU between a detection and a track is below this value, they are not associated.

    car_counter = 0
    # cars_set = set()
    crossed_ids = {}

    current_max_id = -1

    while True:

        success, frame = cap.read()
        frame = cv2.resize(frame, (const.VIDEO_WIDTH_PIXELS, const.VIDEO_HEIGHT_PIXELS))

        shaded_frame = cv2.bitwise_and(frame, area_mask)
        yolo_results = yolo_model(shaded_frame, stream=True)  # yolo applied on the shaded frame
        detections = np.empty((0, 5))  # 5 columns, currently 0 rows, to stack them later
        # each row is a detected object

        cv2.line(frame, (const.LIMIT_LINE['x1'], const.LIMIT_LINE['y1']),
                 (const.LIMIT_LINE['x2'], const.LIMIT_LINE['y2']), (0, 0, 255), thickness=5)

        for result in yolo_results:
            # typically it's just one  result per frame but because it's a Python generator we have to loop through
            boxes = result.boxes
            for box in boxes:
                # get  each coordinates of every bounding box
                x1, y1, x2, y2 = box.xyxy[0]  # 0 indexed even tho it's the only tuple
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                box_width = x2 - x1
                box_height = y2 - y1

                # confidence
                confidence = box.conf[0]
                # class
                cls_name = const.YOLO_CLASSES[int(box.cls[0])]

                if cls_name in const.TO_BE_DETECTED_VEHICLES and confidence > const.CONFIDENCE_THRESHOLD:  # Only detect vehicles
                    # box applied on the original frame
                    # cvzone.cornerRect(frame,(x1,y1,box_width,box_height,))
                    # put text on frame
                    # cvzone.putTextRect(frame,f"{cls_name} {confidence:.2f}",(max(const.MIN_TEXT_X,x1),max(const.MIN_TEXT_Y,y1)),1,1)

                    current_array = np.array([x1, y1, x2, y2, confidence])
                    detections = np.vstack((detections, current_array))

        tracker_results = tracker.update(detections)  # tracker object keeps track of all tracked objects

        for result in tracker_results:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

            box_width = x2 - x1
            box_height = y2 - y1

            cvzone.cornerRect(frame, (x1, y1, box_width, box_height,), colorR=(255, 0, 0))
            cvzone.putTextRect(frame, f"{id}", (max(const.MIN_TEXT_X, x1), max(const.MIN_TEXT_Y, y1)), 1, 1)
            cvzone.putTextRect(frame, f"Car Count: {car_counter}", (20, 30), 2, 1)

            # draw the circle
            cx, cy = x1 + box_width // 2, y1 + box_height // 2,
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # check if center passed the line
            if const.LIMIT_LINE['x1'] <= cx and cx <= const.LIMIT_LINE['x2'] \
                    and const.LIMIT_LINE['y1'] - 20 <= cy and cy <= const.LIMIT_LINE['y2'] + 20:

                # check if car already crossed and registered in the Dict
                if not id in crossed_ids:
                    car_counter += 1
                    crossed_ids[id] = True


        cv2.imshow(const.FRAME_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
