import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0) #For webcam, webcam mac dinh la 0
# cap.set(3,1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/people.mp4")  # test video o day

model = YOLO("../Yolo-weights/yolov8n.pt")  # thay doi phien ban yolo o day, co cac phien ban n(nano), l(large), m(medium)

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
              ]  # nhan dien do vat, muon them thi them vao sau

mask = cv2.imread("mask.png")#ve vung nhan dien, co the ve tren canva

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)#co the tu chinh bang tay de xem hieu ung

limitsUp = [103, 161, 296, 161] #chinh do dai va vi tri dong ke xac nhan
limitsDown = [527, 489, 735, 489] #chinh do dai va vi tri dong ke xac nhan

totalCountUp = [] #list, de tranh truong hop dem phai 1 phan tu 2 lan
totalCountDown = [] #list, de tranh truong hop dem phai 1 phan tu 2 lan

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))#chinh vi tri cua dong ke nhan dien
    results = model(imgRegion, stream=True)#import vung nhan dien vao chuong trinh

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:  # hien thi duong vien ben ngoai vat the
            x1, y1, x2, y2 = box.xyxy[0]  # xac dinh 4 canh cua hinh vuong bao quanh vat the
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10)  # chinh mau chinh o day(Ctrl + click conerReact)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if ( #o vuong ngoai dung de theo doi doi tuong
                    currentClass == "person") and conf > 0.3:
                # chinh doi tuong xet duoc o day
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(40, y1)),
                #                    scale=0.6, thickness=1,
                #                    offset=3)  # chinh scale, thickness cua phan chu mieu ta o day
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])  # tao mang de theo doi
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    # cv2.line(img,(limits[0], limits[1]),(limits[2],limits[3]),(0, 0, 255), 5) #dong ke, thay doi gia tri cua dong ke o day
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result#id la so thu tu
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img,(x1, y1, w, h), l=9, rt=2,colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(40, y1)),
                           scale=2, thickness=3,
                           offset=10)  # chinh scale, thickness cua phan chu mieu ta o day, chinh thu muon xem o text:


        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[1] + 20: #vung nhan dien cua vach ke. co the tang giam gia tri 20
            if totalCountUp.count(id) == 0:# dem 1 phan tu dung 1 lan
                totalCountUp.append(id)#them 1 phan tu vao count
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5) #vach mau xanh la cay khi nhan

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[1] + 20: #vung nhan dien cua vach ke. co the tang giam gia tri 20
            if totalCountDown.count(id) == 0:# dem 1 phan tu dung 1 lan
                totalCountDown.append(id)#them 1 phan tu vao count
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5) #vach mau xanh la cay khi nhan

    #cvzone.putTextRect(img, f'Count: {len(totalCountUp)}', (50, 50)) #o doan Count dung resultsTracker de dem so nguoi
                                                                            #trong phong, con dung totalCount de dem nguoi qua vach
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1200, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion) #hien thi mang loc anh
    cv2.waitKey(1) #0 de co the pause vid, 1 de cho vid chay tu nhien
