'''
  overlay_geoinfo_on_image.py
'''

import os
import glob
import carlos_utils.image_utils as imgu
import carlos_utils.geolocation_utils as geou
import carlos_utils.file_utils as fu

baseFolder = '/Users/carlos.aguilar/Documents/pictures/pictures abuelo'
glob_pattern = os.path.join(baseFolder, '*.HEIC')
pic_files = glob.glob(glob_pattern)


for idx, this_file_path in enumerate(pic_files):

  exif_info = imgu.get_EXIF_info_from_file(this_file_path)
  location_info = geou.reverse_coordinates_geopy(exif_info['latitude_dec'], exif_info['longitude_dec'])

  text = geou.get_text_location(location_info) 
  
  img_1_data = imgu.load_heic_image_as_numpy(this_file_path)

  img_text = imgu.overlay_text(img_1_data, text)
  imgu.show_image(img_text)







# one file


idx = 2
this_file_path = pic_files[idx]
exif_info = imgu.get_EXIF_info_from_file(this_file_path)

location_info = geou.reverse_coordinates_geopy(exif_info['latitude_dec'], exif_info['longitude_dec'])
fu.printJSON(location_info)
print(geou.get_text_location(location_info))
