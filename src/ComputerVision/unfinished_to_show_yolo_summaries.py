'''
unfinished_to_show_yolo_summaries.py
'''


# To delete
import cv2

filepath = '/Volumes/CarlosPictures/iPhone/2020/JPG/labels (yolo v3)/json/objects_in_all_images.json'
info_images = fu.readJSONFile(filepath)
#info_images.keys()

laptop_images = info_images['laptop']

for img_path in laptop_images:

  image = cv2.imread(img_path)
  cv2.imshow("Image", image)

img_path = laptop_images[1]
image = cv2.imread(img_path)
[fPath, fName] = os.path.split(img_path)



cv2.imshow(imagename, image)



import matplotlib.pyplot as plt
imgu.show_imagefile(img_path)

# which is implemented as


import cv2
import matplotlib.pyplot as plt

def load_image(img_path, useCV2=False):
  '''
    Load an image given the correct path.
    Both methods return a uint8 'numpy.ndarray' but the CV2 intensities are not 
    directly compatible with matplotlib.
  '''
  if useCV2:
    this_image = cv2.imread(img_path)
  else:
    this_image = plt.imread(img_path)

  return this_image

def show_image(this_image):
  '''
    Show image using matplotlib
  '''
  fig, ax = plt.subplots()
  im = ax.imshow(this_image)
  ax.axis('off')
  plt.show()


def show_imagefile(img_path):
  show_image(load_image(img_path))