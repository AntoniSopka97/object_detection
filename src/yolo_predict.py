import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import os
from src.sort import Sort
from loguru import logger
import random

class ObjectDetections:
    
    def __init__(self, capture_index):
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        logger.info(f'Use device  {self.device}')
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
    
    def load_model(self):
        model = YOLO("./weight/yolov8m.pt")
        model.fuse()
        logger.info(f'model {model}')
        return model
    
    def predict(self, frame):
        results = self.model(frame, verbose = False)
        return results
    
    def get_results(self, results):
        detections_list = []
        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidences[0]] #, class_id
            detections_list.append(merged_detection)
        return np.array(detections_list)
    
    def draw_id(self, img, bboxes, ids):
        for bbox, id_ in zip(bboxes, ids):
            x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.putText(img, f"ID: {str(id_)}",
                        (x1,y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        return img
    
    def draw_bounding_boxes_id(self, frame, results):
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, clss in zip(boxes, classes):
            if clss != -1:
                random.seed(int(clss))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(
                    frame,
                    f"{self.CLASS_NAMES_DICT[clss]}",
                    (x1,y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
        return frame
    
    
    def __call__(self, stframe, kpi_text):
        
        cap = cv2.VideoCapture(self.capture_index)
        if isinstance(self.capture_index, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        sort = Sort(max_age= 20,
                    min_hits= 8,
                    iou_threshold=0.5)
        
        while cap.isOpened():
            
            start_time = time.perf_counter()
            
            ret, frame = cap.read()
            if ret:
                results = self.predict(frame)
                detections_list = self.get_results(results)
                
                if len(detections_list) == 0:
                    detections_list = np.empty((0, 5))
        
                res = sort.update(detections_list)
                boxes_track = res[:,:-1]
                boxes_ids = res[:,-1].astype(int)

                frame = self.draw_id(frame, boxes_track, boxes_ids)
                frame = self.draw_bounding_boxes_id(frame, results)

                resize_frame = cv2.resize(frame, (0,0), fx = 0.5, fy=0.5, interpolation=cv2.INTER_AREA)    
                fps = 1/(time.perf_counter() - start_time)
                logger.info(f'FPS {fps}')
                stframe.image(resize_frame, channels = 'BGR', use_column_width = True)
                kpi_text.write(f"FPS: {fps}")
            else:
                break
