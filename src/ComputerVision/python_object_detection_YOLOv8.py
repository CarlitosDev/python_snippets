'''

	https://github.com/ultralytics/ultralytics



	Documentation:
	https://docs.ultralytics.com/

	python_object_detection_YOLOv8.py

	source ~/.bash_profile && pip3 install ultralytics


	10.01.2023 - First attempt.
	
'''
import torch
torch.__version__

import torchvision
torchvision.__version__


img_path = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/Photos/calendar/both-IMG_3680.JPG'

from ultralytics import YOLO, checks as ultra_checks
ultra_checks()

img_path ='https://ultralytics.com/images/zidane.jpg'

# Detection
detection_models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m']
detection_model = detection_models[0].lower() + '.pt'
model = YOLO(detection_model)# load a pretrained YOLOv8* model
model.predict(source=img_path)

# Detection train COCO
# model.train(data='coco128.yaml', epochs=1)  # train the model
# model.predict(source=img_path)  # predict on an image


# Segmentation
segmentation_models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m']
segmentation_model = segmentation_models[0].lower() + '-seg.pt'
yolo_segf_model = YOLO(segmentation_model)# load a pretrained YOLOv8* model
yolo_segf_model.predict(source=img_path)

#Classification
classification_models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m']
classification_model = classification_models[2].lower() + '-cls.pt'
yolo_cls_model = YOLO(classification_model)# load a pretrained YOLOv8* model
cls_result = yolo_cls_model.predict(source=img_path)

aa = '/Users/carlos.aguilar/Documents/EF_repos/data_science_utilities/src/yolov8m-cls.pt'




model.train(data="coco128.yaml")  # train the model
model.val()  # evaluate model performance on the validation set
model.predict(source=img_path)  # predict on an image
model.export(format="onnx")  # export the model to ONNX format




import torch
torch.__version__

import torchvision
# torchvision.__version__
# torchvision.__file__

# Model
# yolov5s is the smallest
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)




# with Mateo's picture
import cv2
img_path = '/Users/carlos.aguilar/Google Drive/to_print.jpg'
# Inference
# load our input image and grab its spatial dimensions
image = cv2.imread(img_path)

results = model(image)

# Results
results.print()
results.display()

results.show()

df_results = results.pandas().xyxy[0]
df_results.shape



import carlos_utils.file_utils as fu
fu.to_random_excel_file(df_results)




## TODO: workout how to use the Torch model with videos


# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, PIL, OpenCV, numpy, multiple

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.




##
# Use yolov5 on stram video
