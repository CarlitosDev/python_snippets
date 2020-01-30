import os
import pyocr
import pyocr.builders
from PIL import Image



# Simple image to string
imFolder    = '/Users/carlos.aguilar/Documents/Beamly/VMUA-Hollition/Holition SDK/latest Hol-SDK analysis'
currentFile = 'carlos.png';
customerFilePath = os.path.join(imFolder, currentFile)


tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(customerFilePath), builder=pyocr.builders.DigitBuilder())

print(text)