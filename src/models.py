import torch
from super_gradients.training import models
from loguru import logger
import cv2
from src.coco_class_name import className
import numpy as np
import math


DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

class Models:
    def __init__(self) -> None:
        self.model = self._load_model()
        

    def _load_model(self):
        self.model  = models.get(model_name='yolo_nas_s',
                    pretrained_weights= "coco").to(DEVICE)
        
        logger.info(f'Model is load')
        
    def _predict_from_image(self, image, confidence, st):
        if self.model == None:
            self.model = self._load_model()
        result = list(self.model.predict(image, conf = confidence, fuse_model = False))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, label) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            class_name = className[int(label) + 1]
            conf = math.ceil((confidence * 100))/100
            label = f'{class_name}{conf}'
            cv2.rectangle(image, (x1,y1), (x2,y2), [0,255,255], 3)
            cv2.putText(image, label, (x1, y1-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
            st.subheader('Output Image')
            st.image(image, channels = 'BGR', use_column_width = True)