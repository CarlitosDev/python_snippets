'''
issues_installing_opencv.py
'''


pip3 install opencv-python
python3 -c "import cv2; print(cv2.__version__)"


4.6.0.66
AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)

from here: https://github.com/opencv/opencv-python/issues/591

_registerMatType is a part of OpenCV 4.5.4
Please ensure that you have installed one version of OpenCV only.


pip3 uninstall opencv-python
pip3 uninstall opencv-contrib-python

pip3 install opencv-python
4.6.0