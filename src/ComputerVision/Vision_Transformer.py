'''
	
	
	Vision Transformer (ViT) model pre-trained on ImageNet-21k 
	(14 million images, 21,843 classes) at resolution 224x224, 
	and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes)
	at resolution 224x224. 
	It was introduced in the paper 
	"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"


	https://huggingface.co/google/vit-base-patch16-224


	Model hub:
	https://huggingface.co/models?search=google/vit


	I understand it aims to summarise one picture with a word, but unfortunately
	it doesn't do a good job.

'''


from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# this one says it's a powerdrill??
img_path = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750/_DSC9686.JPG'

#Predicted class: seat belt, seatbelt
img_path ='/Users/carlos.aguilar/Documents/ComputerVision/face recognition tester/Mateo/IMG_9886.jpg'

#Predicted class: teddy, teddy bear
img_path ='/Users/carlos.aguilar/Documents/ComputerVision/face recognition tester/Mateo/IMG_1695.JPG'

# Predicted class: ping-pong ball ??
img_path ='/Users/carlos.aguilar/Documents/ComputerVision/face recognition tester/labelled/Kath.JPG'


image = Image.open(img_path)
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])