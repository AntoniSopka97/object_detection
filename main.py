import streamlit as st
from loguru import logger
from src.detection import predict_from_video, predict_from_image
from src.yolo_predict import ObjectDetections
from PIL import Image
import cv2
import numpy as np
import tempfile

def main():

    st.title('Object detections with YOLO-NAS')
    st.sidebar.title('Settings')
    st.sidebar.subheader('Parameters')

    app_mode = st.sidebar.selectbox('Choose the App Mode',
                                    ['About App', 
                                     'Run on Image',
                                     'Run on Video',
                                     'Run on Video YOLOv8'])
    # About App
    if app_mode == 'About App':
        st.markdown('In this project i am using ***YOLO-NAS*** to do Object Detection on Images and Videos.')
        image = np.array(Image.open('data/test.png'))
        st.image(image)
    if app_mode == 'Run on Image':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)
        logger.info(f'Confidence: {confidence}')
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type= ['jpg','png','jpeg'])
        DEMO_IMAGE = './data/you.jpg'
        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text('Original Image')
        st.sidebar.image(image)
        predict_from_image(image=img, confidence=confidence, st=st)

    if app_mode == 'Run on Video':
        st.sidebar.markdown('---')
        use_web_cam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload an Image", type= ['mp4'])
        
        DEMO_VIDEO = './data/20230705_003647.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        if not video_file_buffer:
            if use_web_cam:
                tffile.name = 0
            else:
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_vid.read())
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_vid.read())
        stframe = st.empty()
        kpi1_text = st.markdown("0")
        predict_from_video(video_name= tffile.name, kpi_text=kpi1_text, stframe=stframe)
                
    if app_mode == 'Run on Video YOLOv8':
        st.sidebar.markdown('---')
        use_web_cam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload an Image", type= ['mp4'])
        
        DEMO_VIDEO = './data/20230705_003647.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        if not video_file_buffer:
            if use_web_cam:
                tffile.name = 0
            else:
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_vid.read())
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_vid.read())
        stframe = st.empty()
        kpi1_text = st.markdown("0")
        detector = ObjectDetections(capture_index=tffile.name)
        detector(stframe, kpi1_text)
        
          

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
