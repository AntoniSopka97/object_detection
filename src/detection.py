import torch
import cv2
from super_gradients.training import models
from src.coco_class_name import className
import numpy as np
import math
import time
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
print("The device is: ", DEVICE)
print(DEVICE)


MODEL = models.get(model_name='yolo_nas_s',
                    pretrained_weights= "coco").to(DEVICE)


def predict_from_video(video_name, kpi_text, stframe):
    cap = cv2.VideoCapture(video_name)
    resize_frame = None
    while True:
        ret, frame = cap.read()
        if ret:
            start_time = time.perf_counter()
            result = list(MODEL.predict(frame, conf = 0.35, fuse_model = False))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, label) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                class_name = className[int(label) + 1]
                conf = math.ceil((confidence * 100))/100
                label = f'{class_name}{conf}'
                cv2.rectangle(frame, (x1,y1), (x2,y2), [0,255,255], 3)
                cv2.putText(frame, label, (x1, y1+20), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                resize_frame = cv2.resize(frame, (0,0), fx = 0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if resize_frame is not None:
                stframe.image(resize_frame, channels = 'BGR', use_column_width = True)
                fps = 1/(time.perf_counter() - start_time)
                kpi_text.write(f"FPS: {fps}")
        
        else:
            break

def predict_from_image(image, confidence, st):
    result = list(MODEL.predict(image, conf = confidence, fuse_model = False))[0]
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()
    predicts = []
    for (bbox_xyxy, confidence, label) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_name = className[int(label) + 1]
        conf = math.ceil((confidence * 100))/100
        label = f'{class_name}{conf}'
        predicts.append([class_name, conf, x1,y1,x2,y2])
        cv2.rectangle(image, (x1,y1), (x2,y2), [0,255,255], 3)
        cv2.putText(image, label, (x1, y1-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
    
    st.subheader('Output Image')
    st.image(image, channels = 'BGR', use_column_width = True)
    
    predicts = pd.DataFrame(predicts, columns = [
        'Class', 'Confidence',
        'xtl', 'ytl', 'xbr', 'ybr'
    ])
    st.table(predicts)