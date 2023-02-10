'''

From here:

  Use GANS to create an alter ego looking like Anime characters

  Much of the code comes from here
  https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py

  19.11.2021 - First attempt



'''

import os
import numpy as np
import cv2
import carlos_utils.file_utils as fu
import carlos_utils.image_utils as imgu
import carlos_utils.computer_vision_utils as cvis
import torch
from PIL import Image
# Run this is PyTorch crashes when doing the request
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


model1 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", 
  pretrained="face_paint_512_v1",  device="cpu")

face2paint = torch.hub.load('AK391/animegan2-pytorch:main', 'face2paint', 
  size=512, device="cpu",side_by_side=False)

model2 = torch.hub.load("AK391/animegan2-pytorch:main", "generator",
    pretrained=True,device="cpu",   progress=False
)

face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, device="cpu",side_by_side=False
)


this_file_path = '/Users/carlos.aguilar/Documents/Pictures_Carlos/carlos/IMG_8636.jpg'

folder, fileName, fileExt = fu.fileparts(this_file_path)
image_info = dict(folder=folder,fileName=fileName,fileExt=fileExt)

this_image = imgu.load_image(this_file_path)

imgu.show_image(this_image)


pil_image = Image.fromarray(this_image)

output_model_1 = face2paint(model1, pil_image)
imgu.show_image(output_model_1)

output_model_2 = face2paint(model2, pil_image)
imgu.show_image(output_model_2)
