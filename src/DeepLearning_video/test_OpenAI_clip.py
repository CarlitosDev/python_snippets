'''
  test_OpenAI_clip.py

  Given an image, find the probability of some text to represent the image

'''

# Let's use the model from HuggingFace instead of the official repo (https://github.com/openai/CLIP)


from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from torch import argmax as pt_argmax

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://i1.sndcdn.com/artworks-000030842531-j2oc3y-t500x500.jpg"
image = Image.open(requests.get(url, stream=True).raw)


label_options = ["a photo of a cat", \
  "rifle",
  "a photo of a person",
  "a photo of a guitar", \
  "guitar",
  "a photo of a rock and roll star"]

inputs = processor(text=label_options, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
idx_label = pt_argmax(probs, dim=1).item()

detected_prob = probs[0][idx_label].item()
detected_label = label_options[idx_label]
print(f'CLIP assigns label {detected_label.upper()} a probability of {detected_prob:2.3f}')


# in np
# probs = logits_per_image.softmax(dim=1).detach().numpy()

