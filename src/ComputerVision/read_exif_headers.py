)'''
	read_exif_headers.py

	pip3 install exifread


	(It works for HEIC images too)
'''




import exifread
from fractions import Fraction

import uuid

path_name = '/Users/carlos.aguilar/Documents/house_pics/IMG_0850.JPG'
path_name = '/Users/carlos.aguilar/Documents/temp_carlos/DSC01039.jpg'
path_name = '/Users/carlos.aguilar/Documents/temp_carlos/DSC01026.jpg'
path_name = '/Users/carlos.aguilar/Downloads/IMG_7648.HEIC'

with open(path_name, 'rb') as f:
  tags = exifread.process_file(f)

gps_tags = [itag for itag in tags.keys() if 'gps' in itag.lower()]

for tag in gps_tags:
  print(tag, tags[tag])


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
##from datetime import datetime
#datetime.strptime('02-07-2017', '%d-%m-%Y')
#%Y:%m%d %

import cv2

# load our input image and grab its spatial dimensions
image = cv2.imread(path_name)
(H, W) = image.shape[:2]

# x -> horizontal
x0 = int(W*0.5)
y0 = int(H*0.80)

print(x0,y0)
dy = int(W/30)
text = latitude_tag + '\n' + longitude_tag + '\n' + datetime_tag

lineType = 10
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale  = 5
# Blue color in BGR
fontColor = [0,0,255]

for i, line in enumerate(text.split('\n')):
	y1 = y0 + i*dy
	cv2.putText(image, line, (x0, y1), font, fontScale, fontColor, lineType)
	print(x0,y1)

fext = path_name.split('.')[-1]
postfix = str(uuid.uuid1())[0:4]
new_name = path_name.replace(fext, 'processed.' + postfix + '.' + fext)
cv2.imwrite(new_name, image)
#cv2.imshow("Image", image)
#plt.show()








this_tag = 'GPS GPSDate'
tags[this_tag]

this_tag = 'Image GPSInfo'
tags[this_tag]

tags['EXIF DateTimeOriginal']


for tag in [*tags.keys()]:
	print('\t', tag)




# python3 -m pip install pyexiv2
import pyexiv2
img = pyexiv2.Image(path_name)
data = img.read_exif()
img.close()
gps_tags = [itag for itag in data.keys() if 'gps' in itag.lower()]
for tag in gps_tags:
	print('\t', tag, data[tag])

current_tag = data['Exif.Image.DateTime']
#current_tag.strftime('%A %d %B %Y, %H:%M:%S')







#path_name = '/Users/carlos.aguilar/Documents/temp_carlos/IMG_7531.JPG'

from PIL import Image
import PIL.ExifTags

image = Image.open(path_name)
exif_data_PIL = image._getexif()

for k, v in PIL.ExifTags.TAGS.items():
	if 'GPSInfo' in v:
		 print(exif_data_PIL[k])


exif_data = {}
for k, v in PIL.ExifTags.TAGS.items():
	if k in exif_data_PIL:
		print(exif_data_PIL[k])
		exif_data[v] = {"tag": k, "raw": exif_data_PIL[k], "processed": exif_data_PIL[k]}

exif_data['GPSInfo']['raw'][1]
exif_data['GPSInfo']['raw'][2]



########



import exifread
from fractions import Fraction
import uuid



with open(path_name, 'rb') as f:
	tags = exifread.process_file(f)



# LOCATION_ID: The GPS latitude/longitude coordinates in ISO-6709 format: ±DD.DDDD±DDD.DDDD
pip3 install reverse_geocode
# https://pypi.org/project/reverse_geocoder/
import reverse_geocode
coordinates = (-37.81, 144.96), (31.76, 35.21)
reverse_geocode.search(coordinates)