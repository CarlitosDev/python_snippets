'''

Weight file: it’s the trained model, the core of the algorythm to detect the objects.
Cfg file: it’s the configuration file, where there are all the settings of the algorythm.
Name files: contains the name of the objects that the algorythm can detect.

This script follows 
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/


wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3.cfg


https://github.com/pjreddie/darknet

Yolo_v4
https://github.com/AlexeyAB/darknet


wget https://github.com/pjreddie/darknet/blob/master/data/coco.names


'/Users/carlosAguilar/Documents/yolo3/darknet/cfg/yolov3.cfg'


OpenCV 3.4.2+
cv2.__version__
'4.2.0'

pip3 install opencv-python

python3 '/Users/carlosAguilar/Google Drive/PythonSnippets (1)/python_object_detection_YOLOv3.py'
'''


import numpy as np
import time
import cv2
import os
import json

show_image = True
min_obj_confidence = 0.5
NMS_threshold = 0.3

min_obj_confidence = 0.5
NMS_threshold = 0.3


# Move the folder in my personal laptop to blablab/ComputerVision
#yolo_folder = '~/Documents/yolo3'
yolo_folder = os.path.expanduser('~/Documents/ComputerVision/yolo3')
weights_file = os.path.join(yolo_folder, 'yolov3.weights')
cfg_file = os.path.join(yolo_folder, 'darknet/cfg/yolov3.cfg')
labels_path = os.path.join(yolo_folder, 'coco.names')

img_path = '/Users/carlosAguilar/Documents/Mateo 1 year - album/11 Feb 2k19/IMG_1741.JPG'
img_path = '/Users/carlosAguilar/Documents/2020 - Barbados and Antigua/102ND750/_DSC0689.JPG'
img_path ='/Users/carlosAguilar/Documents/Mateo 1 year - album/13 April 2k19/IMG_3008.JPG'
img_path = '/Users/carlosAguilar/Documents/2020 - Barbados and Antigua/102ND750/_DSC0693.JPG'

img_path = '/Users/carlos.aguilar/Documents/Carlos WallPapers/_DSC4293_edited.JPG'

#net = cv2.dnn.readNetFromDarknet("/home/babu/darknet/cfg/yolov3-tiny.cfg","/home/babu/darknet/yolov3-tiny.weights")

LABELS = open(labels_path).read().strip().split("\n")

# Load Yolo
net = cv2.dnn.readNet(cfg_file, weights_file)

# If yoloV4
yolo_folder = os.path.expanduser('~/Documents/ComputerVision/yolo4')
weights_file = os.path.join(yolo_folder, 'yolov4.weights')
cfg_file = os.path.join(yolo_folder, 'yolov4.cfg')
net = cv2.dnn.readNet(cfg_file, weights_file)
net = cv2.dnn_DetectionModel(cfg_file, weights_file)



# initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# load our input image and grab its spatial dimensions
image = cv2.imread(img_path)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(
    image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > min_obj_confidence:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)


# “non-maxima suppression”
# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_obj_confidence, NMS_threshold)

# ensure at least one detection exists
detected_objects = []
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        d = {'object_id': int(i), 'x': x, 'y': y, 'w': w, 'h': h,
            'object': LABELS[classIDs[i]], 'confidence': confidences[i]}
        detected_objects.append(d)
        if show_image:
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2)


# show the output image
if show_image:
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    cv2.imwrite("image_processed.png", image)


print(json.dumps(detected_objects))
