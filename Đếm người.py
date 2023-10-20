import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture(1) #For webcam, webcam mac dinh la 0
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/people.mp4") #test video o day


model = YOLO("../Yolo-weights/yolov8n.pt")#thay doi phien ban yolo o day, co cac phien ban n(nano), l(large)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "wireless headphone"
              ]# nhan dien do vat, muon them thi them vao sau

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)#co the tu chinh bang tay de xem hieu ung


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes: #hien thi duong vien ben ngoai vat the
            x1, y1, x2, y2 = box.xyxy[0] #xac dinh 4 canh cua hinh vuong bao quanh vat the
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (  # o vuong ngoai dung de theo doi doi tuong
                 currentClass == "person") and conf > 0.6:
                cvzone.cornerRect(img, (x1, y1, w, h))  # chinh mau chinh o day(Ctrl + click conerReact)
                currentArray = np.array([x1, y1, x2, y2, conf])  # tao mang de theo doi
                detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result  # id la so thu tu
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            #cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(40, y1)),scale=2, thickness=3, offset=10)  # chinh scale, thickness cua phan chu mieu ta o day, chinh thu muon xem o text:

        cv2.putText(img, f"So nguoi: {str(len(resultsTracker))}", (0, 100), cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7) #chinh scale, thickness cua phan chu mieu ta o day
    cv2.imshow("Image", img)
    cv2.waitKey(1)
