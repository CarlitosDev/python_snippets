'''
  Some methods for images


  Requires:  
  
  brew install libffi libheif
  pip3 install pyheif

'''

import utils.image_utils as imu
import cv2
import matplotlib.pyplot as plt
import pyheif
from PIL import Image
import os

# CV2 is not working. No idea why?
# img_path = '/Users/carlos.aguilar/Documents/temp carlos pics Mateo/101ND750/_DSC9677.JPG'
#image = cv2.imread(img_path)
#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



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


def save_image(image_data, img_path: str, useCV2=True):
  """Write image to disk using CV2

  Args:
      image_data ([type]): [description]
      img_path (str): [description]
      useCV2 (bool, optional): [description]. Defaults to True.
  """  
  cv2.imwrite(img_path, image_data)

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




def load_heic_image(img_path):
  '''
    Read HEIC images
  '''

  with open(img_path, 'rb') as f:
      data = f.read()

  pyheif_img = pyheif.read_heif(data)
  img_data = Image.frombytes(mode=pyheif_img.mode, \
    size=pyheif_img.size, data=pyheif_img.data)

  return img_data


def load_heic_image_as_numpy(img_path: str):
  """Load a HEIC image as a numpy array

    Recommended to fiddle around with PyTorch

  Args:
      img_path (str): fullpath to the HEIC file

  Returns:
      [type]: nd-array
  """  

  return np.asarray(load_heic_image(img_path))


def from_heic_to_jpg(img_1_path, img_output=None):
  
  img_1_data = load_heic_image(img_1_path)

  if not img_output:
    img_output = img_1_path.replace('HEIC', 'jpg')

  img_1_data.save(img_output, format="JPEG")
  print(f'JPG file saved to {img_output}')




def extract_frames_from_video(video_path, num_frames_to_capture = 10):

  foldername, filename, ext = fu.fileparts(video_path)
  cap = cv2.VideoCapture(video_path)

  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration_seconds = frame_count/fps

  frame_interval = int(frame_count/num_frames_to_capture)

  frame_counter = 0

  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      if frame_counter%frame_interval == 0:
          frame_name = os.path.join(foldername, f'filename_frame_{frame_counter}.jpg')
          cv2.imwrite(frame_name, frame)
      frame_counter+=1
      
  cap.release()
  cv2.destroyAllWindows()



def get_EXIF_info_from_file(path_name):
  '''
    Get the EXIF info from a file
    It uses the library exifread
  '''
  from fractions import Fraction
  with open(path_name, 'rb') as f:
    tags = exifread.process_file(f)

  gps_tags = [itag for itag in tags.keys() if 'gps' in itag.lower()]

  lat_ref = str(tags['GPS GPSLatitudeRef'])
  lat_values = str(tags['GPS GPSLatitude']).replace('[','').replace(']','').split(',')

  lat_degrees = int(lat_values[0])
  lat_minutes = int(lat_values[1])
  frc = Fraction(lat_values[-1])
  lat_seconds = frc.numerator/frc.denominator
  latitude_tag = f'''{lat_degrees} {lat_minutes}\' {lat_seconds}\'\' {lat_ref}'''



  lon_ref = str(tags['GPS GPSLongitudeRef'])
  lon_values = str(tags['GPS GPSLongitude']).replace('[','').replace(']','').split(',')
  frc = Fraction(lon_values[-1])


  lon_degrees = int(lon_values[0])
  lon_minutes = int(lon_values[1])
  lon_seconds = frc.numerator/frc.denominator
  longitude_tag = f'''{lon_degrees} {lon_minutes}\' {lon_seconds}\'\' {lon_ref}'''

  datetime_tag = str(tags['Image DateTime'])


  exif_info  = {'latitude_tag': latitude_tag,
  'lat_ref':lat_ref,
  'lat_values':lat_values,
  'lat_degrees':lat_degrees,
  'lat_minutes':lat_minutes,
  'lat_seconds':lat_seconds,
  'longitude_tag': longitude_tag,
  'lon_ref':lon_ref,
  'lon_values':lon_values,
  'lon_degrees':lon_degrees,
  'lon_minutes':lon_minutes,
  'lon_seconds':lon_seconds,
  'datetime_tag':datetime_tag}

  return exif_info