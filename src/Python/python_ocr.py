# Good tutorial for Tesseract
# https://nanonets.com/blog/ocr-with-tesseract/#gettingboxesaroundtext


import os
from PIL import Image




# This package works pretty well actually
#pip3 install -U git+https://github.com/madmaze/pytesseract.git
import pytesseract

imFolder    = '/Users/carlos.aguilar/Documents/temp'
currentFile = 'for_OCR.png'
filePath = os.path.join(imFolder, currentFile)
text = pytesseract.image_to_string(Image.open(filePath))
print(text)

import pyperclip
pyperclip.copy(text)




'''

The PyOCR relies on OCR libraries that might now be installed in the system. 
Check out with the command:
tools = pyocr.get_available_tools()
If empty, go to a terminal and do some brew magic.
brew install tesseract

'''

import pyocr
import pyocr.builders


# Simple image to string
imFolder    = '/Users/carlos.aguilar/Documents/Beamly/VMUA-Hollition/Holition SDK/latest Hol-SDK analysis'
currentFile = 'carlos.png';
customerFilePath = os.path.join(imFolder, currentFile)


tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(customerFilePath), builder=pyocr.builders.DigitBuilder())

print(text)



# Let's try to read some code from images. This doesn't work well
imFolder    = '/Users/carlos.aguilar/Documents/temp'
currentFile = 'for_OCR.png'
filePath = os.path.join(imFolder, currentFile)
tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(filePath), builder=pyocr.builders.DigitBuilder())



