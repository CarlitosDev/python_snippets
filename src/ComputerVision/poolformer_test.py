'''

  PoolFormer is a model that replaces attention token mixer in transfomrers with extremely simple operator, pooling.


'''

import carlos_utils.image_utils as imgu
from transformers import PoolFormerFeatureExtractor, PoolFormerForImageClassification

feature_extractor = PoolFormerFeatureExtractor.from_pretrained('sail/poolformer_m48')
model = PoolFormerForImageClassification.from_pretrained('sail/poolformer_m48')

img_path = '/Users/carlos.aguilar/Downloads/IMG_3453.jpg'
pil_image = imgu.load_image_as_PIL(img_path)


inputs = feature_extractor(images=pil_image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# 
_, indices = logits.sort(descending=True)

num_classes = 5


# model predicts one of the 1000 ImageNet classes
for idx in range(0, num_classes):
  predicted_class_idx = indices[0][idx].item()
  probability_class_idx = logits[0][predicted_class_idx].item()
  predicted_class = model.config.id2label[predicted_class_idx]
  print(f"Predicted class {predicted_class} with p={probability_class_idx:3.2f}")




# other sources...

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

extractor = AutoFeatureExtractor.from_pretrained("sail/poolformer_m48")
model = AutoModelForImageClassification.from_pretrained("sail/poolformer_m48")


###
from transformers import PoolFormerFeatureExtractor, PoolFormerForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = PoolFormerFeatureExtractor.from_pretrained('sail/poolformer_m48')
model = PoolFormerForImageClassification.from_pretrained('sail/poolformer_m48')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])




