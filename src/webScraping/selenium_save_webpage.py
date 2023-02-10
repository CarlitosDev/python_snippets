
import os
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
import json



chromiumFolder = '~/Documents/chromedriver/'
baseFile   = 'chromedriver'
driverFile = os.path.expanduser(os.path.join(chromiumFolder, baseFile))
driver = webdriver.Chrome(driverFile)



url_to_download = 'https://www.ef.co.uk/'
driver.get(url_to_download)


# option A
subFolder = '/Users/carlos.aguilar/Documents/tempRubbish'
htmlFile   = os.path.join(subFolder, 'class_details.html')
with open(htmlFile, 'w') as f:
    f.write(driver.page_source)

