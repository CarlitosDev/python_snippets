'''

This script follows 
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

python3 '/Users/carlos.aguilar/Google Drive/PythonSnippets/ComputerVision/run_YOLOv3_folder.py'
'''


import json
import os
import cv2
import time
import numpy as np

baseFolder = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750'
baseFolder = '/Volumes/CarlosPictures/Photos (all)/2k19-Mauritius/102ND750'
baseFolder = '/Volumes/CarlosPictures/Photos (all)/2k18_Cuba - fotos/5_stars'
baseFolder = '/Volumes/CarlosPictures/Photos (all)/2k18_Cuba - fotos/4_stars'

baseFolder = '/Users/carlosAguilar/Documents/2020 - Barbados and Antigua/102ND750'
baseFolder = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750'

baseFolder = '/Volumes/CarlosPictures/iPhone_latest_dump/iColours/dump_August_2k20/iColours/2019/JPG'
baseFolder = '/Volumes/CarlosPictures/iPhone_latest_dump/dump_August_2k20/iColours/2020/JPG'


min_obj_confidence = 0.5
NMS_threshold = 0.3

yolo_folder = os.path.expanduser('~/Documents/ComputerVision/yolo3')
weights_file = os.path.join(yolo_folder, 'yolov3.weights')
cfg_file = os.path.join(yolo_folder, 'darknet/cfg/yolov3.cfg')
labels_path = os.path.join(yolo_folder, '..', 'coco.names')


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


def load_yolo():
    net = cv2.dnn.readNet(cfg_file, weights_file)
    coco_labels = open(labels_path).read().strip().split("\n")
    return net, coco_labels


def yolo_detect(net, image, min_obj_confidence):
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
    return boxes, confidences, classIDs


def yolo_NMS(boxes, confidences, min_obj_confidence, NMS_threshold, coco_labels, classIDs, image):
    # “non-maxima suppression”
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, min_obj_confidence, NMS_threshold)
    # ensure at least one detection exists
    detected_objects = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            d = {'object_id': int(i), 'x': x, 'y': y, 'w': w, 'h': h,
                 'object': coco_labels[classIDs[i]], 'confidence': confidences[i]}
            detected_objects.append(d)
            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colour_set[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(
                coco_labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        2, color, 2)
    return detected_objects, image


if __name__ == "__main__":

    object_dict = {}

    net, coco_labels = load_yolo()
    # initialize a list of colors to represent each possible class label
    colour_set = np.random.randint(
        0, 255, size=(len(coco_labels), 3), dtype="uint8")

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Create the output folders
    labelled_pics_folder = os.path.join(baseFolder, 'yolov3_labelled_pics')
    makeFolder(labelled_pics_folder)
    labels_folder = os.path.join(baseFolder, 'yolov3_labels')
    makeFolder(labels_folder)
    objects_folder = os.path.join(baseFolder, 'yolov3_labels_objects')
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
            boxes, confidences, classIDs = yolo_detect(net, image, min_obj_confidence)

            detected_objects, image = yolo_NMS(
                boxes, confidences, min_obj_confidence, NMS_threshold, coco_labels, classIDs, image)

            json_filename = os.path.join(labels_folder, filename + '.json')
            writeJSONFile(json.dumps(detected_objects), json_filename)

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
    writeJSONFile(json.dumps(object_dict), json_filename)