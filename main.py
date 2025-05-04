from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import constants as const


def main():

    cap = cv2.VideoCapture(const.CARS_VIDEO_PATH)
    cap.set(const.WIDTH_ID, const.VIDEO_WIDTH_PIXELS)
    cap.set(const.HEIGHT_ID, const.VIDEO_HEIGHT_PIXELS)

    yolo_model = YOLO(const.YOLO_PATH)

    area_mask = cv2.imread(const.CARS_AREA_MASK_PATH)

    while True:

        success, frame = cap.read()
        frame = cv2.resize(frame, (const.VIDEO_WIDTH_PIXELS, const.VIDEO_HEIGHT_PIXELS))

        shaded_frame = cv2.bitwise_and(frame,area_mask)


        yolo_results = yolo_model(frame, stream=True)  # frame yolo results

        for result in yolo_results:
            # typically it's just one  result per frame but because it's a Python generator we have to loop through

            boxes = result.boxes

            for box in boxes:

                # get  each coordinates of every bounding box
                x1,y1,x2,y2 = box.xyxy[0] # 0 indexed even tho it's the only tuple
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                box_width = x2-x1
                box_height = y2-y1
                # confidence
                confidence = box.conf[0]
                # class
                cls_name = const.YOLO_CLASSES[int(box.cls[0])]

                if cls_name in const.TO_BE_DETECTED_VEHICLES and confidence > const.CONFIDENCE_THRESHOLD: # Only detect vehicles

                    cvzone.cornerRect(frame,(x1,y1,box_width,box_height,))
                    # put text on frame
                    cvzone.putTextRect(frame,f"{cls_name} {confidence:.2f}",(max(const.MIN_TEXT_X,x1),max(const.MIN_TEXT_Y,y1)),1,1)

        # cv2.imshow(const.FRAME_TITLE, frame)
        cv2.imshow('shaded',shaded_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
