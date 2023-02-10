'''
scene_understanding.py



'''

import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

this_file_path = '/Users/carlos.aguilar/Desktop/Screenshot 2022-07-18 at 21.10.33.png'

this_file_path ='/Users/carlos.aguilar/Downloads/New Folder With Items/IMG_6321.HEIC'

import carlos_utils.image_utils as imgu

this_PIL_image = imgu.load_image_as_PIL(this_file_path).convert("RGB")
this_PIL_image = imgu.load_image_as_PIL(this_file_path).convert("RGB")

this_PIL_image = imgu.load_heic_image_as_numpy(this_file_path)
image = feature_extractor(images=this_PIL_image, return_tensors="pt").pixel_values.to(device)
clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
caption_ids = model.generate(image, max_length = max_length)[0]
caption_text = clean_text(tokenizer.decode(caption_ids))
print(caption_text)