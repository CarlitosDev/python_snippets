

'''

using_facebook_detectron2.py




To manipulate the videos
brew install ffmpeg



cd /Users/carlos.aguilar/Documents/Detectron2

git clone https://github.com/facebookresearch/detectron2
cd detectron2
source ~/.bash_profile && python3 setup.py install
# or
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html


# get one of our videos
wget -O MOB_15.4.1.1.1.mp4 "https://qa-et.ef-cdn.com//juno/50/23/65/v/502365/MOB_15.4.1.1.1.mp4" 


Cut 30 seconds starting at 20
ffmpeg -ss 00:00:20 -i MOB_15.4.1.1.1.mp4 -to 00:00:30 -c copy output.mp4






# Note: this is currently BROKEN due to missing codec. See https://github.com/facebookresearch/detectron2/issues/2901 for workaround.

run detectron2/demo/demo.py --config-file detectron/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input output.mp4 --confidence-threshold 0.6 --output video-output.mkv \
  --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl


source ~/.bash_profile && python3 detectron2/demo/demo.py --config-file detectron/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input output.mp4 --confidence-threshold 0.6 --output video-output.mkv \
  --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl


source ~/.bash_profile && python3 demo/demo.py --config-file detectron/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input output.mp4 --confidence-threshold 0.6 --output video-output.mkv \
  --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl



source ~/.bash_profile && python3


'''


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")


# to run detectron on single images
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


img_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/IMG_6143.jpg'
img_path ='/Users/carlos.aguilar/Documents/house_pics/IMG_0852.JPG'
# load our input image and grab its spatial dimensions
image = cv2.imread(img_path)
outputs = predictor(image)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out_image = out.get_image()[:, :, ::-1]
cv2.imshow("Image", out_image)
cv2.imwrite("image_processed.png", out_image)





# PANOPTIC 
# Inference with a panoptic segmentation model
img_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/IMG_6104.jpg'
# load our input image and grab its spatial dimensions
image = cv2.imread(img_path)

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)


outputs = predictor(image)
panoptic_seg, segments_info = outputs["panoptic_seg"]
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

out_image = out.get_image()[:, :, ::-1]
# cv2.imshow("Image", out_image)


# 'translate' the segments info

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
for sinfo in segments_info:
  category_idx = sinfo["category_id"]
  if sinfo['isthing']:
    text = metadata.thing_classes[category_idx]
  else:
    text = metadata.stuff_classes[category_idx]
  sinfo['object'] = text

import carlos_utils.file_utils as fu
fu.printJSON(segments_info)  



print(panoptic_seg["instances"].pred_classes)

cv2.imwrite("panoptic_image_processed.png", out_image)





#####
def extract_frames_from_video(video_path, num_frames_to_capture = 10):

  foldername, filename, ext = fu.fileparts(video_path)
  cap = cv2.VideoCapture(video_path)

  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration_seconds = frame_count/fps

  frame_interval = int(frame_count/num_frames_to_capture)

  frame_counter = 0

  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      if frame_counter%frame_interval == 0:
          frame_name = os.path.join(foldername, f'filename_frame_{frame_counter}.jpg')
          cv2.imwrite(frame_name, frame)
      frame_counter+=1
      
  cap.release()
  cv2.destroyAllWindows()





###########################
# PANOPTIC segmentaion

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)



num_frames_to_capture = 10

video_path = '/Users/carlos.aguilar/Documents/EF_Content/EFxAWS/lumiere_data/BBC_videos/BBC_Commuting (1).mp4'
foldername, filename, ext = fu.fileparts(video_path)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = frame_count/fps

frame_interval = int(frame_count/num_frames_to_capture)

frame_counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if frame_counter%frame_interval == 0:

      outputs = predictor(frame)
      panoptic_seg, segments_info = outputs["panoptic_seg"]
      v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
      out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

      out_image = out.get_image()[:, :, ::-1]
      
      frame_name = os.path.join(foldername, f'{filename}_segmented_frame_{frame_counter}.jpg')
      cv2.imwrite(frame_name, out_image)

    frame_counter+=1

cap.release()
cv2.destroyAllWindows()