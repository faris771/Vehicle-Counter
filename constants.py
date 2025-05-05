CARS_VIDEO_PATH = 'videos/cars.mp4'
WIDTH_ID, HEIGHT_ID, = 3, 4
YOLO_PATH = 'yolo_weights/yolov8n.pt'
VIDEO_WIDTH_PIXELS = 1080
VIDEO_HEIGHT_PIXELS = 608
FRAME_TITLE = 'Live Footage'
MIN_TEXT_X = 0
MIN_TEXT_Y = 40
CONFIDENCE_THRESHOLD = 0.2

OUTPUT_PATH = 'output/result.avi'


LIMIT_LINE = {
    'x1': 250,
    'y1': 297,
    'x2': 550,
    'y2': 297
}

CARS_AREA_MASK_PATH = 'masks/cars_area_mask.png'

TO_BE_DETECTED_VEHICLES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
