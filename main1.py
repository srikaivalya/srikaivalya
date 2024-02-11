import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)
  
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('busfinal.mp4')

with open("coco.txt", "r", encoding="utf-8") as my_file:
    data = my_file.read()
class_list = data.split("\n") 

count = 0

area1 = [(259, 488), (281, 499), (371, 499), (303, 466)]

tracker = Tracker()

counter = []

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
   
    results = model.predict(frame)
    
    for result in results.xyxy:
        a = result.data
        px = pd.DataFrame(a).astype("float")
        
        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])
            bbox_idx = tracker.update(list)
            for id, rect in bbox_idx.items():
                x3, y3, x4, y4 = rect
                cx = x3
                cy = y4
                result = cv2.pointPolygonTest(np.array(area1), ((cx, cy)), False)
                if result >= 0:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if id not in counter:
                    counter.append(id)
        cv2.polylines(frame, [np.array(area1)], True, (255, 0, 0), 2)
        p = len(counter)
        cvzone.putTextRect(frame, f'Counter:-{p}', (50, 60), 2, 2)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
