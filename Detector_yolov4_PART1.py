# Object detection using YOLO V4
# Detector_yolov4_PART1.py
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

from yolov4.tf import YOLOv4

# Initializing YOLOv4
yolo = YOLOv4()

# setting 'COCO' class names
yolo.classes = "files/coco.names"

# Creating model
yolo.make_model()

# Initializing model to pre-trained state
yolo.load_weights("files/yolov4.weights", weights_type="yolo")

inp = int(input('Choose the format for detecting objects : \n 1.Image \n 2.Video \n'))

if inp == 1: #for image
	yolo.inference(media_path="data/image.jpg")

elif inp == 2:#for video
	yolo.inference(media_path="data/video.mp4", is_image=False)
