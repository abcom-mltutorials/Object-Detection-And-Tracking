# Object detection using YOLO V3
# Detector_yolov3_PART1.py

import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

# we are using the class names from coco dataset,
# which has 80 different classes

classnames = []

with open('files/coco.names') as f:
    classnames = f.read().rstrip('\n').split('\n')

nnet = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')

#Uncomment following line if you have GPU
#nnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCL)

#setting the confidence and nms Threshold
conf_thresh = 0.29
nms_thresh = 0.3

def findObjects(outputs, img):
    #getting the dimensions of the original image
    target_height, target_width = img.shape[:2]
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for d in output:
            scores = d[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_thresh:
                w,h = int(d[2]*target_width), int(d[3]*target_height)
                x,y = (int(d[0]*target_width) - w/2), (int(d[1]*target_height) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,conf_thresh,nms_thresh)
    print(f'Number of detected objects: {len(indices)}')

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        #print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img, f'{classnames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

inp = int(input('Choose the input source for object detection: \n 1.Image \n 2.Video \n 3.Webcam \n'))

if inp == 1: #for image
    # We are using yolo v3 320 version,
    # which takes images of size 320x320
    input_size = 320
    image = cv2.imread('data/image01.jpg')
    # display original image
    cv2.imshow('Image',image)

    # input to YOLO network is in blob format
    blob_img = cv2.dnn.blobFromImage(image,1/255,(input_size,input_size),[0,0,0],1,crop=False)
    nnet.setInput(blob_img)

    # get a list of layers in the network
    layerNames = nnet.getLayerNames()

    # get names of output layers
    outputNames = [layerNames[i[0]-1] for i in nnet.getUnconnectedOutLayers()]
    
    print (outputNames)

    # forward outputs of output layers to extract
    # information on detected objects
    objectInfo = nnet.forward(outputNames)
    
    # mark all detected objects
    findObjects(objectInfo, image)
    
    # objectInfo is a list of arrays, where each row
    # contains information on detected objects
    # print the dimensions of the three outputs
    print (objectInfo[0].shape)
    print (objectInfo[1].shape)
    print (objectInfo[2].shape)
    
    # display revised image
    cv2.imshow('Image',image)
    
    # wait for user to quit
    cv2.waitKey(0)

elif inp == 2: #for video
    video = cv2.VideoCapture('data/video01.mp4')

    input_size = 320

    # iterating over each frame in the video,
    # and detecting objects in each frame
    while True:
        success, img = video.read()

        blob_img = cv2.dnn.blobFromImage(img,1/255,(input_size,input_size),[0,0,0],1,crop=False)
        nnet.setInput(blob_img)

        layerNames = nnet.getLayerNames()

        outputNames = [layerNames[i[0]-1] for i in nnet.getUnconnectedOutLayers()]

        outputs = nnet.forward(outputNames)

        findObjects(outputs, img)

        cv2.imshow('Vid',img)

        # each frame is shown for 1 millisecond,
        # and until the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

elif inp == 3: #for webcam
    video = cv2.VideoCapture(0)
    input_size = 320

    while True:
        success, img = video.read()
        blob_img = cv2.dnn.blobFromImage(img,1/255,(input_size,input_size),[0,0,0],1,crop=False)
        nnet.setInput(blob_img)

        layerNames = nnet.getLayerNames()

        outputNames = [layerNames[i[0]-1] for i in nnet.getUnconnectedOutLayers()]

        outputs = nnet.forward(outputNames)

        findObjects(outputs, img)

        cv2.imshow('Vid',img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

else:
    print('Wrong Input')
