# Vehicle Detection and Tracking System

This project implements a vehicle detection and tracking system using YOLOv8 for object detection and the SORT algorithm for tracking. The system processes video footage to detect and track vehicles, count them as they cross a predefined line, and display the results in real-time.

## Features
- **Object Detection**: Uses YOLOv8 to detect vehicles such as cars, trucks, buses, motorcycles, and bicycles.
- **Object Tracking**: Tracks detected vehicles across frames using the SORT algorithm.
- **Vehicle Counting**: Counts vehicles as they cross a specified line in the video.
- **Real-Time Visualization**: Displays bounding boxes, unique IDs, and the vehicle count on the video feed.

## Requirements
- Python 3.8 or higher
- OpenCV
- NumPy
- cvzone
- ultralytics (for YOLOv8)
- scikit-image (for SORT algorithm)
- 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/faris771/Car_Counter.git
   cd vehicle-tracking