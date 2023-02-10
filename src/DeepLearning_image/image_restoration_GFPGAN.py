'''

From here:

  
  Much of the code comes from here
  https://huggingface.co/spaces/akhaliq/GFPGAN/blob/main/app.py

  
   - First attempt

   Requirements:
   pip3 install gfpgan


  wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth /Users/carlos.aguilar/Documents/deep_learning_models/GFPGAN/GFPGANCleanv1-NoCE-C2.pth




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


from gfpgan import GFPGANer

model_path = '/Users/carlos.aguilar/Documents/deep_learning_models/GFPGAN/GFPGANCleanv1-NoCE-C2.pth'

bg_upsampler = None

# set up GFPGAN restorer
restorer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler)






this_file_path = '/Users/carlos.aguilar/Documents/photo_restoration/baby_pater.jpeg'

folder, fileName, fileExt = fu.fileparts(this_file_path)
image_info = dict(folder=folder,fileName=fileName,fileExt=fileExt)

this_image = imgu.load_image(this_file_path)

imgu.show_image(this_image)




cropped_faces, restored_faces, restored_img = restorer.enhance(this_image, \
  has_aligned=False, only_center_face=False, paste_back=True)

from PIL import Image
restored_image = Image.fromarray(restored_faces[0][:,:,::-1])
imgu.show_image(restored_image)
