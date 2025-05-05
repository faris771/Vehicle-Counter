# 🚗 Vehicle Detection and Tracking System

A side project crafted to **master computer vision** and software engineering practices. This system detects, tracks, and counts vehicles in real-time using **YOLOv8** for object detection and **SORT** for multi-object tracking — all built with a clean, maintainable architecture that follows the **SOLID principles**.

![Preview](output/result.gif)

---

## ✨ Key Features

- 🧠 **Smart Object Detection**  
  Detects cars, trucks, buses, motorcycles, and bicycles using YOLOv8 with high accuracy and speed.

- 🎯 **Reliable Multi-Object Tracking**  
  Uses the SORT algorithm to consistently track vehicles across video frames, even with partial occlusions.

- 🔢 **Accurate Vehicle Counting**  
  Counts each vehicle once as it crosses a virtual counting line, preventing duplicates.

- 🖼️ **Real-Time Visualization**  
  Renders bounding boxes, unique IDs, and the live count directly on the video feed.

- 🧼 **Clean Code with SOLID Principles**  
  The codebase is modular, maintainable, and adheres to best practices like separation of concerns and object-oriented design.

---

## 📦 Requirements

- Python 3.8 or higher  
- OpenCV  
- NumPy  
- cvzone  
- ultralytics (`YOLOv8`)  
- scikit-image (used by SORT)  
- filterpy  

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/faris771/Car_Counter.git
   cd vehicle-tracking

2. Install dependencies:
 ```bash
   pip install -r requirements.txt
 ```
3. Run the script:
```bash
   python3 main.py
 ```
## 🧠 Motivation
This project was developed as part of my journey to master computer vision fundamentals and improve my Python architecture skills. It merges machine learning, real-time systems, and clean software engineering.

## 📬 Contact

Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/faris-abufarha/) or open an issue if you have feedback or suggestions!
