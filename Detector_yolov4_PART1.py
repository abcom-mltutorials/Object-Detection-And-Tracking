
#================================================
Copyright @ 2020 **ABCOM Information Systems Pvt. Ltd.** All Rights Reserved.

Licensed under the Apache Licaense, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

#================================================


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
	yolo.inference(media_path="data/image00.jpg")

elif inp == 2:#for video
	yolo.inference(media_path="data/video00.mp4", is_image=False)
