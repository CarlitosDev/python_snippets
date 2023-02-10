'''

  background_removal.py

'''

import carlos_utils.image_utils as imgu
import os
import cv2



# one file
this_file_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/Luca/IMG_8477.HEIC'
exif_info = imgu.get_EXIF_info_from_file(this_file_path)




this_algo = 'MOG2'
this_algo = ''
if this_algo == 'MOG2':
  backSub = cv2.createBackgroundSubtractorMOG2()
else:
  backSub = cv2.createBackgroundSubtractorKNN()



this_image = imgu.load_heic_image_as_numpy(this_file_path)
fgMask = backSub.apply(this_image)

cv2.rectangle(this_image, (10, 2), (100,20), (255,255,255), -1)
## [display_frame_number]

## [show]
#show the current frame and the fg masks
imgu.show_image(this_image)

imgu.show_image(fgMask)



# pip3 install cvzone mediapipe
from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()
# RGB
background_colour = (0,0,255)
imgOut = segmentor.removeBG(this_image, background_colour, threshold=0.8)
imgu.show_image(imgOut)




# Using mediapipe directly
# https://google.github.io/mediapipe/
# YouTube: https://www.youtube.com/c/MediaPipe

import cv2
import mediapipe as mp
import numpy as np

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
results = selfie_segmenter.process(this_image)

results = selfie_segmenter.process(cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB))

# Generate solid color images for showing the output selfie segmentation mask.
fg_image = np.zeros(this_image.shape, dtype=np.uint8)
fg_image[:] = MASK_COLOR
bg_image = np.zeros(this_image.shape, dtype=np.uint8)
bg_image[:] = BG_COLOR
condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
output_image = np.where(condition, fg_image, bg_image)

imgu.show_image(output_image)



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


rawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh =  mp_face_mesh.FaceMesh(static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

results = face_mesh.process(cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB))
annotated_image = this_image.copy()

for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())

imgu.show_image(annotated_image)



# myself
/Users/carlos.aguilar/Documents/Pictures_Carlos/carlos/IMG_8636.HEIC