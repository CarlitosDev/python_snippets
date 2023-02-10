'''

  
  
  https://qa-et.ef-cdn.com/juno/50/59/27/v/505927/UK_1.2.3.5.1.mp4

'''

import carlos_utils.image_utils as imgu
import os
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# myself
# one file
this_file_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/carlos/IMG_8636.jpg'
# this_image = imgu.load_heic_image_as_numpy(this_file_path)
this_image = imgu.load_image(this_file_path, useCV2=False)


# First, remove the background
# segmentor = SelfiSegmentation()
# background_colour = (255,255,255)
# imgOut = segmentor.removeBG(this_image, background_colour, threshold=0.6)
# imgu.show_image(imgOut)


rawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh =  mp_face_mesh.FaceMesh(static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

# results = face_mesh.process(cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
results = face_mesh.process(cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB))




# default
tesselation_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
# a bit custom
mesh_colour = mp_drawing_styles._GREEN
mesh_thickness = 2
tesselation_style = mp_drawing.DrawingSpec(color=mesh_colour, thickness=mesh_thickness)


# customise the face contours
# default
face_mesh_contours_style = mp_drawing_styles.get_default_face_mesh_contours_style()
# customised
from mediapipe.python.solutions import face_mesh_connections
_THICKNESS_CONTOURS = 4
_FACEMESH_CONTOURS_CONNECTION_STYLE = {
    face_mesh_connections.FACEMESH_LIPS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._RED, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYE:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._PURPLE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYEBROW:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._YELLOW, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYE:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._PURPLE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._YELLOW, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_FACE_OVAL:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._RED, thickness=_THICKNESS_CONTOURS)
}
face_mesh_contours_style = {}
for k, v in _FACEMESH_CONTOURS_CONNECTION_STYLE.items():
  for connection in k:
    face_mesh_contours_style[connection] = v


# default
face_mesh_iris_connections_style = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
# custom
face_mesh_iris_connections_style = {}
left_spec = mp_drawing.DrawingSpec(color=mp_drawing_styles._PEACH, thickness=_THICKNESS_CONTOURS)
for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
  face_mesh_iris_connections_style[connection] = left_spec
right_spec = mp_drawing.DrawingSpec(color=mp_drawing_styles._PEACH, thickness=_THICKNESS_CONTOURS)
for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
  face_mesh_iris_connections_style[connection] = right_spec



annotated_image = this_image.copy()
for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      # facemesh
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=tesselation_style
          )
      # face contours (includes lips and eyebrows)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=face_mesh_contours_style)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=face_mesh_iris_connections_style)

imgu.show_image(annotated_image)

