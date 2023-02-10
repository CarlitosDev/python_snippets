'''
convert_HEIC_to_jpg.py
'''


import utils.image_utils as imu
import os
import utils.file_utils as fu



baseFolder = os.path.expanduser('~/Documents/temp_carlos/Mateo and Luca')

# Find files
import glob
glob_pattern = os.path.join(baseFolder, '*.HEIC')
pic_files = glob.glob(glob_pattern)
for idx, this_file_path in enumerate(pic_files):
  img_1_data = imu.load_heic_image(this_file_path)  
  img_1_jpg_path = img_1_path.replace('HEIC', 'jpg')
  img_1_data.save(img_1_jpg_path, format='JPEG')


'''
1-
Face Verification - Demo
This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or based64 encoded images is also welcome.
'''


img_1_path = os.path.join(baseFolder, 'IMG_2737.HEIC')
img_2_path = os.path.join(baseFolder, 'IMG_6476.HEIC')


img_2_data = imu.load_heic_image(img_2_path)


img_2_jpg_path = img_2_path.replace('HEIC', 'jpg')


img_2_data.save(img_2_jpg_path, format="JPEG")
