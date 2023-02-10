'''

OpenCV and Yolo_v4

https://github.com/AlexeyAB/darknet


This script follows https://github.com/opencv/opencv/pull/17185 which is very neat
example on YoloV4


Installation:
cd '/Users/carlos.aguilar/Documents/computerVision/yolov4'
wget 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'

# Do this manually as wget downloads the whole html
wget 'https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg'








Remainder:
  Weight file: it’s the trained model, the core of the algorythm to detect the objects.
  Cfg file: it’s the configuration file, where there are all the settings of the algorithm.
  Name files: contains the name of the objects that the algorythm can detect.



OpenCV
cv2.__version__
'4.4.0'

python3 '/Users/carlosAguilar/Google Drive/PythonSnippets/python_object_detection_YOLOv4.py'

Updates:
  03.09.2020 - First version

'''


import numpy as np
import time
import cv2
import os
import json

show_image = True

min_obj_confidence = 0.1
NMS_threshold = 0.3


img_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/IMG_6143.jpg'

# Move the folder in my personal laptop to blablab/ComputerVision
yolo_folder = os.path.expanduser('~/Documents/computerVision/yolov4')
weights_file = os.path.join(yolo_folder, 'yolov4.weights')
cfg_file = os.path.join(yolo_folder, 'yolov4.cfg')

# read the labels
labels_path = os.path.join(yolo_folder, '..',  'coco.names')
coco_labels = open(labels_path).read().strip().split("\n")


# Load yoloV4
# based on https://github.com/opencv/opencv/pull/17185
net = cv2.dnn_DetectionModel(cfg_file, weights_file)
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)



# load our input image and grab its spatial dimensions
image = cv2.imread(img_path)
(H, W) = image.shape[:2]


start = time.time()
classes, confidences, boxes = net.detect(image, confThreshold=min_obj_confidence, nmsThreshold=NMS_threshold)
end = time.time()
print("YOLO v4 took {:.6f} seconds".format(end - start))

# For the boxes and labels
colour_set = np.random.randint(0, 255, size=(len(coco_labels), 3), dtype="uint8")

object_id = -1
detected_objects = []

for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

  object_id+=1
  object_name = coco_labels[classId]
  
  label_txt = f'{object_name}: {confidence:3.2f}'

  # box
  left, top, width, height = box
  x,y,w,h = box

  d = {'object_id': object_id, 'x': x, 'y': y, 'w': w, 'h': h,
      'object': object_name, 'confidence': confidence}
  detected_objects.append(d)

  labelSize, baseLine = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
  
  top = max(top, labelSize[1])
  cv2.rectangle(image, box, color=(0, 255, 0), thickness=3)
  cv2.rectangle(image, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
  
  color = [int(c) for c in colour_set[classId]]
  cv2.putText(image, label_txt, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

# show the output image
if show_image:
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    cv2.imwrite("image_processed.png", image)


print(detected_objects)
print(json.dumps(detected_objects, default=str))