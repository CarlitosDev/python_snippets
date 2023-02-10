'''
  pip3 install transformers --upgrade
  
  https://huggingface.co/microsoft/trocr-base-printed
'''
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import carlos_utils.image_utils as imgu


model_sizes = ['base', 'large']
model_size = model_sizes[1]
current_model = f'microsoft/trocr-{model_size}-printed'

processor = TrOCRProcessor.from_pretrained(current_model)
model = VisionEncoderDecoderModel.from_pretrained(current_model)





this_file_path = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/end of lesson recap/exampleTimAckroyd.png'
this_PIL_image = imgu.load_image_as_PIL(this_file_path).convert("RGB")
#this_PIL_image.size
# image_grayscale = this_PIL_image.convert('L')
# np.array(image_grayscale).shape

imgu.show_image(this_PIL_image)

pixel_values = processor(images=this_PIL_image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)


# this is not working...
# internally the model scales the image (https://github.com/microsoft/unilm/blob/master/trocr/pic_inference.py#L34)
# im = Image.open(this_file_path).convert('RGB').resize((384, 384))
# imgu.show_image(im)




# official example
# load image from the IAM database (actually this model is meant to be used on printed text)
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
pixel_values_wc = processor(images=image, return_tensors="pt").pixel_values
# (248, 76)
imgu.show_gray_image(image)

generated_ids_wc = model.generate(pixel_values_wc)
generated_text = processor.batch_decode(generated_ids_wc, skip_special_tokens=True)[0]
print(generated_text)