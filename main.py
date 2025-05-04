from ultralytics import YOLO
import cv2
import cvzone
import numpy as np




CARS_VIDEO_PATH = 'videos/cars.mp4'
WIDTH_ID, HEIGHT_ID,  = 3,4


def main():

    cap = cv2.VideoCapture(CARS_VIDEO_PATH)
    cap.set(WIDTH_ID, 720)
    cap.set(HEIGHT_ID, 480)

    while True:

        success, frame = cap.read()
        cv2.imshow("Live Footage", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break


if __name__ == '__main__':

    main()

