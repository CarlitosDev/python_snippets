'''

	https://github.com/ultralytics/yolov5

	Documentation:
	https://docs.ultralytics.com/

	python_object_detection_YOLOv5.py

	source ~/.bash_profile && python3 -m pip install torchvision -U
	source ~/.bash_profile && python3 -m pip install torch --upgrade


	python3 -c 'import urllib.request; urllib.request.urlopen("https://pypi.org")'

	


	cd '/Users/carlos.aguilar/Documents/repos_and_software'
	git clone https://github.com/pytorch/vision.git
	cd vision
	MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python3 setup.py install
	cd ..
	python3 -c 'import torchvision as tv; print(tv.__version__)'

'''

'''
	Extras:
	How to use it directly in AWS:
	https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart
'''


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
