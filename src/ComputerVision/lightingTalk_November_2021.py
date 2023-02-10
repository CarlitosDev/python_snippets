'''
lightingTalk_November_2021.py


A picture is worth...a multitude of analyses

'''

# Python's annoying imports
import os
import numpy as np
import cv2
import carlos_utils.file_utils as fu
import carlos_utils.image_utils as imgu
import carlos_utils.computer_vision_utils as cvis
import carlos_utils.geolocation_utils as geou



'''
  0 - Read a HEIC image

  
  From https://www.macworld.co.uk/feature/what-is-heic-3660408/:
  
  "HEIC is the file format name Apple has chosen for the new HEIF standard. HEIF s
  tands for High Efficiency Image Format, and, as the name suggests, 
  is a more streamlined way of storing image files. Using advanced, 
  modern compression methods it allows digital photographs to be
  created in smaller files sizes, all while retaining higher image 
  quality than its JPEG alternative."

  Requirements:
    pip3 install pyheif
  

'''

this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/IMG_8923.HEIC'
this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/Victoria_station.HEIC'
# this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/HighamHillPark.HEIC'
# this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/IMG_9175.HEIC'

folder, fileName, fileExt = fu.fileparts(this_file_path)
image_info = dict(folder=folder,fileName=fileName,fileExt=fileExt)

this_image = imgu.load_heic_image_as_numpy(this_file_path)

imgu.show_image(this_image)


'''
  1 - Parse picture information from the EXIF headers
'''


'''
  (1.a) - Camera maker and model and camera settings

  From https://www.wikiwand.com/en/Exif:
  Exchangeable image file format is a standard that specifies the formats for images,
  and ancillary tags used by digital cameras (including smartphones).


'''
camera_info, camera_settings = imgu.get_EXIF_info_from_file(this_file_path)
fu.printJSON(camera_info)
image_info['camera_info'] = camera_info

fu.printJSON(camera_settings)
image_info['camera_settings'] = camera_settings





'''
  (1.B) - Parse the geographical information contained in the EXIF headers

  >> EXIF-Geolocation

  The Exif format has standard tags for location information.
  Mobile phones have a built-in GPS receiver that stores the location 
  information in the Exif header when a picture is taken.

  Requirements:
    pip3 install geopy

'''
# 1.b.1 - GPS info from EXIF
exif_gps_info = imgu.get_GPS_EXIF_info_from_file(this_file_path)
fu.printJSON(exif_gps_info)
image_info['exif_gps_info'] = exif_gps_info

# 1.b.2 - From coordinates to location
location_info = geou.reverse_coordinates_geopy(exif_gps_info['latitude_dec'], exif_gps_info['longitude_dec'])
fu.printJSON(location_info)
image_info['location_info'] = location_info





'''
  3 - What does my picture contain?
'''



'''
  2.1 - Object detection  
  Requirements:
    pip3 install --upgrade torchvision torch 

  Use YoloV5 by Utralytics:
	  https://docs.ultralytics.com/
  * there's controversy about this release of Yolo

  See the available models in https://pytorch.org/hub/ultralytics_yolov5/

  Organise the objects per class, ie: the key 'chair' 
  will be a list of N detected items.

'''
import torch
import torchvision
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
results = model(this_image)
results.print()
# results.display()
results.show()

df_results = results.pandas().xyxy[0]
_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence']

detected_objects = {}
for idx, iRow in df_results.iterrows():
  detected_objects.setdefault(iRow['name'], []).append(iRow[_cols].to_dict())

fu.printJSON(detected_objects)
image_info['detected_objects'] = detected_objects

objects_and_quantity = {k: len(v) for k,v in detected_objects.items()}
fu.printJSON(objects_and_quantity)





'''
  2.2 - Panoptic segmentation (Detectron2 by Facebook)
  
  Requirements:
    pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
    OR manual install:
    cd /Users/carlos.aguilar/Documents/Detectron2
    git clone https://github.com/facebookresearch/detectron2
    (or git pull)
    cd detectron2
    source ~/.bash_profile && python3 setup.py install

'''
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()

# to run detectron on single images
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'


# switch models easily
# cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")
predictor = DefaultPredictor(cfg)

outputs = predictor(this_image)
panoptic_seg, segments_info = outputs["panoptic_seg"]
v = Visualizer(this_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

out_image = out.get_image()[:, :, ::-1]
imgu.show_image(out_image)


# 'translate' the segments info
total_area = float(this_image.shape[0]*this_image.shape[1])

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
for sinfo in segments_info:
  category_idx = sinfo["category_id"]
  sinfo['percentage_area'] = sinfo['area']/total_area

  if sinfo['isthing']:
    text = metadata.thing_classes[category_idx]
  else:
    text = metadata.stuff_classes[category_idx]
  sinfo['object'] = text

fu.printJSON(segments_info)
image_info['segmented_objects'] = segments_info



'''
Pytesseract

pip3 install pytesseract --upgrade

brew install tesseract
(tesseract --version)
pip3 install pyocr --upgrade


pip3 install pytesseract pyocr --upgrade
'''
import pytesseract


# img_rgb = cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_rgb))
from PIL import Image
this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/coffee_shop.jpg'
pil_image = Image.open(this_file_path)

this_image_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
imgu.show_image(this_image_grey)

text = pytesseract.image_to_string(pil_image)
print(text)

boxes = pytesseract.image_to_boxes(this_image, lang='eng')
print(boxes)

_boxes = [list(map(int, i)) for i in [b.split(" ")[1:-1] for b in boxes.split("\n")]]
_boxes.remove([])

# res = np.zeros_like(self.gray_img
bounded = this_image.copy()
img_h, img_w, _ = bounded.shape

for box in _boxes:
  b = (int(box[0]), int(img_h - box[1]), int(box[2]), int(img_h - box[3]))
  cv2.rectangle(bounded, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

imgu.show_image(bounded)

tesseract_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(this_image, lang='eng', config=tesseract_config)
print(text)


custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(this_image, config=custom_config)
print(text)


import pyocr
tools = pyocr.get_available_tools()
from PIL import Image
text = tools[0].image_to_string(pil_image, builder=pyocr.builders.DigitBuilder())
print(text)





def full_OCR(self):
  bounded = self.img.copy()
  res = np.zeros_like(self.gray_img)

  string = pytesseract.image_to_string(Image.open(self.image_file))
  if string == u'':
      return bounded, res

  boxes = pytesseract.image_to_boxes(Image.open(self.image_file))
  boxes = [map(int, i) for i in [b.split(" ")[1:-1] for b in boxes.split("\n")]]

  for box in boxes:
    b = (int(box[0]), int(self.h - box[1]), int(box[2]), int(self.h - box[3]))
    cv2.rectangle(bounded, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    cv2.rectangle(res, (b[0], b[1]), (b[2], b[3]), 255, -1)

  return bounded, res 



'''
pip3 install transformers --upgrade


To use these methods, install directly from GitHub, not the PIP.
pip3 install git+https://github.com/huggingface/transformers.git
(transformers-4.13.0)


source ~/.bash_profile && AWS_PROFILE=efdata-qa && python3 -c "from transformers import TrOCRProcessor;processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')"
source ~/.bash_profile && AWS_PROFILE=efdata-qa && python3 -c "from transformers import TrOCRProcessor;processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')"
'''


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# load image from the IAM database (actually this model is meant to be used on printed text)
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw)


processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
pixel_values = processor(images=this_image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)





'''
  Image classification

  From here https://huggingface.co/google/vit-base-patch16-224:
  Use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes.
 
  BibTeX refs:
    @misc{wu2020visual,
        title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
        author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
        year={2020},
        eprint={2006.03677},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

  Requires:
  pip3 install transformers

'''

from transformers import ViTFeatureExtractor, ViTForImageClassification

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=this_image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print(f'This image is classified as {model.config.id2label[predicted_class_idx].upper()}')




'''
  A few more things worth checking....

  pip3 install mediapipe
'''
import mediapipe as mp

this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/Mateo_badass.jpg'
folder, fileName, fileExt = fu.fileparts(this_file_path)
mateo_image_info = dict(folder=folder,fileName=fileName,fileExt=fileExt)

# mateo_image = imgu.load_heic_image_as_numpy(this_file_path)
mateo_image = imgu.load_image(this_file_path, useCV2=False)
imgu.show_image(mateo_image)




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



# import carlos_utils.computer_vision_utils as cvu
# mesh_results = cvu.get_facemesh_from_image(mateo_image)
# mesh_image = cvu.generate_mesh_image(mateo_image, mesh_results)
# imgu.show_image(mesh_image)








'''

  Image embeddings


  Adapted from here: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

'''

## >> see /Volumes/GoogleDrive/My Drive/PythonSnippets/ComputerVision/tester_img_to_vec.py


import torch
import torchvision.models as models
model = models.resnet50(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

import torchvision.transforms as transforms
from torch.autograd import Variable

model.eval()

# torch_tensor = torch.from_numpy(this_image).long()
pil_adaptor = transforms.ToPILImage()

# scaler = transforms.Resize((224, 224))
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

a = pil_adaptor(this_image)
b = scaler(a)
c = to_tensor(b)
d = normalize(c)
t_img = Variable(d.unsqueeze(0))


# 3. Create a vector of zeros that will hold our feature vector
#    The 'avgpool' layer has an output size of 512
# original
# my_embedding = torch.zeros(512)
layer_output_size=512
# git hub fella
my_embedding = torch.zeros(len(this_image), layer_output_size, 1, 1)
# 4. Define a function that will copy the output of a layer
def copy_data(m, i, o):
    my_embedding.copy_(o.data)
# 5. Attach that function to our selected layer
h = layer.register_forward_hook(copy_data)
# 6. Run the model on our transformed image
model(t_img)
# 7. Detach our copy function from the layer
h.remove()

my_embedding
