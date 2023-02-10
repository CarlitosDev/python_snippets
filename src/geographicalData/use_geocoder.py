'''
use_geopy.py


 source ~/.bash_profile && pip3 install reverse_geocoder

'''

import reverse_geocoder as rg
coordinates = (29,-84.1),(37,-125) #Both located in the ocean
rg.search(coordinates) 


import carlos_utils.file_utils as fu

fu.printJSON(exif_info)



import re
import utf8

dms2dec('''48°53'10.18"N''')
dms2dec('''2°20'35.09"E''')


dms_str = '''36°41'14.6"N'''
dms2dec(dms_str)
# 36.68738888888888
dms_str = '''6°25'9.0"W'''
dms2dec(dms_str)
#-6.419166666666667


pic_coordinates = (36.68738888888888, -6.419166666666667)
geo_info = rg.search(pic_coordinates)



def dms_to_dec(_degrees, _minutes,_seconds, _ref):
  sign = -1 if re.search('[swSW]', _ref) else 1
  return sign * (int(_degrees) + float(_minutes) / 60 + float(_seconds) / 3600)



dms_to_dec(exif_info['lat_degrees'],exif_info['lat_minutes'],\
  exif_info['lat_seconds'],exif_info['lat_ref'])

dms_to_dec(exif_info['lon_degrees'],exif_info['lon_minutes'],\
  exif_info['lon_seconds'],exif_info['lon_ref'])






def dms2dec(dms_str):
  """Return decimal representation of DMS

  >>> dms2dec(utf8(48°53'10.18"N))
  48.8866111111F

  >>> dms2dec(utf8(2°20'35.09"E))
  2.34330555556F

  >>> dms2dec(utf8(48°53'10.18"S))
  -48.8866111111F

  >>> dms2dec(utf8(2°20'35.09"W))
  -2.34330555556F

  """

  dms_str = re.sub(r'\s', '', dms_str)

  sign = -1 if re.search('[swSW]', dms_str) else 1

  numbers = [*filter(len, re.split('\D+', dms_str, maxsplit=4))]

  degree = numbers[0]
  minute = numbers[1] if len(numbers) >= 2 else '0'
  second = numbers[2] if len(numbers) >= 3 else '0'
  frac_seconds = numbers[3] if len(numbers) >= 4 else '0'

  second += "." + frac_seconds
  return sign * (int(degree) + float(minute) / 60 + float(second) / 3600)