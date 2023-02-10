'''
	pip3 install pdf2image
  (it needs external libs)
'''

import requests
import pdf2image
import pytesseract

pdf = requests.get('https://ieeexplore.ieee.org/ielx7/6287639/9312710/09363114.pdf')
doc = pdf2image.convert_from_bytes(pdf.content)

# Get the article text
article = []
for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(page_data).encode("utf-8")
    # Sixth page are only references
    if page_number < 6:
      article.append(txt.decode("utf-8"))
article_txt = " ".join(article)