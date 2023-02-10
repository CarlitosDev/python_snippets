'''

OpenCV and Yolo_v4

UPDATES: In my personal tests, it does not overperform YOLOV3!



https://github.com/AlexeyAB/darknet
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

This script follows https://github.com/opencv/opencv/pull/17185 which is very neat
example on YoloV4


Remainder:
  Weight file: it’s the trained model, the core of the algorythm to detect the objects.
  Cfg file: it’s the configuration file, where there are all the settings of the algorythm.
  Name files: contains the name of the objects that the algorythm can detect.



OpenCV (cv2.__version__ >> '4.4.0')

Updates:
  03.09.2020 - First version

Notes: 
Based on the former implementation using yolo v3 
(python3 '~./Google Drive/PythonSnippets/ComputerVision/run_YOLOv3_folder.py')

Runner:
python3 '/Users/carlos.aguilar/Google Drive/PythonSnippets/ComputerVision/run_YOLOv4_folder.py'


'''

import json
import os
import cv2
import time
import numpy as np


baseFolder = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750'


show_image = True
min_obj_confidence = 0.5
NMS_threshold = 0.3

# Set the paths
yolo_folder = os.path.expanduser('~/Documents/ComputerVision/yolo4')
weights_file = os.path.join(yolo_folder, 'yolov4.weights')
cfg_file = os.path.join(yolo_folder, 'yolov4.cfg')


def makeFolder(thisPath):
    if not os.path.exists(thisPath):
        os.makedirs(thisPath)


def fileparts(thisPath):
    [fPath, fName] = os.path.split(thisPath)
    [file, ext] = os.path.splitext(fName)
    return fPath, file, ext


def writeJSONFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)


def load_yolov4():
    # load yolov4 and set some parameters
    net = cv2.dnn_DetectionModel(cfg_file, weights_file)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    # read the labels from the COCO dataset
    labels_path = os.path.join(yolo_folder, '..',  'coco.names')
    coco_labels = open(labels_path).read().strip().split("\n")

    return net, coco_labels




def yolov4_detect_objects(net, image, min_obj_confidence, NMS_threshold, coco_labels):

    # Super neat functions for v4
    start = time.time()
    classes, confidences, boxes = net.detect(image, \
      confThreshold=min_obj_confidence, nmsThreshold=NMS_threshold)
    end = time.time()
    print("YOLO v4 took {:.6f} seconds".format(end - start))

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

      return detected_objects, image


if __name__ == "__main__":

    object_dict = {}

    net, coco_labels = load_yolov4()
    # initialize a list of colors to represent each possible class label
    colour_set = np.random.randint(0, 255, \
          size=(len(coco_labels), 3), dtype="uint8")


    # Create the output folders
    labelled_pics_folder = os.path.join(baseFolder, 'yolov4_labelled_pics')
    makeFolder(labelled_pics_folder)
    labels_folder = os.path.join(baseFolder, 'yolov4_labels')
    makeFolder(labels_folder)
    objects_folder = os.path.join(baseFolder, 'yolov4_label_objects')
    makeFolder(objects_folder)

    #for root, dirs, files in os.walk(baseFolder, topdown=True):
    for thisFile in os.listdir(baseFolder):
        
        if '.jpg' in thisFile.lower():
        
            print(f'Processing {thisFile}...', end='')
            img_path = os.path.join(baseFolder, thisFile)
            fPath, filename, ext = fileparts(img_path)

            # load our input image and grab its spatial dimensions
            image = cv2.imread(img_path)
            (H, W) = image.shape[:2]

            detected_objects, image = yolov4_detect_objects(net, image, \
              min_obj_confidence, NMS_threshold, coco_labels)

            json_filename = os.path.join(labels_folder, filename + '.json')
            writeJSONFile(json.dumps(detected_objects, indent=4, default=str), json_filename)

            # Update the objects dictionary
            for idx, iRecord in enumerate(detected_objects):
                val = iRecord.get('object', None)
                if val:
                    object_dict.setdefault(val, []).append(img_path)

            # Save the labelled image
            labelled_filename = os.path.join(
                labelled_pics_folder, filename + '_labelled.' + ext)
            cv2.imwrite(labelled_filename, image)

    # Outlide the main loop
    json_filename = os.path.join(objects_folder,  'objects_in_all_images.json')
    writeJSONFile(json.dumps(object_dict, indent=4, default=str), json_filename)