'''
       Example from here
       https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

'''


import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import os

# This model has 1001 labels?
classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Decode the predictions
labels_filepath = os.path.join(baseFolder, 'ImageNetLabels.txt')
labels_path = tf.keras.utils.get_file(labels_filepath,'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])


el_imagepath = 'https://et2.ef-cdn.com/en-gb/_jcr_content/main-parsys/illustrator_1722217512/image-parsys/image_ee1b.img.jpg/1481621950702.jpg'
baseFolder = '/Users/carlos.aguilar/Documents/EF_Content/ImageAnalysis'
el_img_localpath = '1481621950702.jpg'
el_fullpath = os.path.join(baseFolder, el_img_localpath)

el_image = tf.keras.utils.get_file(el_fullpath, el_imagepath)

# resize to fit the model
original_image = Image.open(el_image)
current_image = original_image.resize(IMAGE_SHAPE)

current_image = np.array(current_image)/255.0
current_image.shape

result = classifier.predict(current_image[np.newaxis, ...])
# 1x1001 
result.shape

predicted_class = np.argmax(result[0], axis=-1)
predicted_class

idxSorted = np.argsort(result[0], axis=-1)[::-1]

# top 5 matches
str_matches = ''
for idx in idxSorted[0:5]:
    predicted_class_name = imagenet_labels[idx]   
    print(predicted_class_name.title())
    str_matches += predicted_class_name.title() +','


plt.imshow(original_image)
plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + str_matches)
plt.show()

