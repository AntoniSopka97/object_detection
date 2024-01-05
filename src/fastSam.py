import sys
sys.path.append('/home/a/project/yolo_nas_object_detection/src/')
sys.path.append('/home/a/project/yolo_nas_object_detection/')
sys.path.append('/home/a/project/yolo_nas_object_detection/FastSAM/')

from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2
import time
from loguru import logger

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = FastSAM('/home/a/project/yolo_nas_object_detection/weight/FastSAM-s.pt')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        start_time = time.perf_counter()
        everything_results = model(
                                    frame,
                                    device=device,
                                    retina_masks=True,
                                    imgsz=512,
                                    conf=0.4,
                                    iou=0.9,
                                    )
        
        # for box in everything_results[0].boxes:
        #     box = box.xyxy.cpu().numpy()[0]
        #     x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #     cv2.rectangle(frame, (x1,y1), (x2,y2), [0,255,255], 3)
        
        resize_frame = cv2.resize(frame, (0,0), fx = 0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        #everything_results = list(everything_results)
        
        prompt_process = FastSAMPrompt(resize_frame, everything_results, device)
        #ann = prompt_process.text_prompt(text='a photo of a person')
        ann = prompt_process.everything_prompt()
        img = prompt_process.plot_to_result(annotations= ann)
        
        fps = 1/ (time.perf_counter() - start_time)
        logger.info(f'FPS {fps}')
        
        cv2.imshow('sam', img)
        
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()  
cap.release()

        