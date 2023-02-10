'''
  webScraping_Duolingo.py

  STEPS:

  1 - Get Chromium from here https://chromedriver.chromium.org/ (*)
    1.a - Execute this on the Chromium file:
    cd '/Volumes/GoogleDrive/My Drive/chromedriver/'
    spctl --add --label 'Approved' chromedriver
  2 - PIP install Selenium
  3 - Python code below


(*) Make sure it matches your current Chrome installation.
For example, check your current version and paste it as
https://chromedriver.storage.googleapis.com/index.html?path=98.0.4758.80/

'''



import os
import pandas as pd
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


# caller method
def get_CEFR_Duolingo(transcription, driver, waitingTime=5):
  '''
    Get the CEFR estimation and breakdown from Duolingo's tool
  '''

  url_duolingo = 'https://cefr.duolingo.com/'
  driver.get(url_duolingo)

  transcription_box = driver.find_element_by_tag_name('textarea')
  transcription_box.send_keys(transcription)
  transcription_box.send_keys(Keys.ENTER)

  #driver.implicitly_wait(8) # seconds
  time.sleep(waitingTime)

  cefr_results = driver.find_element_by_class_name('RZxzk').text


  cefr_results_breakdown = driver.find_element_by_class_name('_2vA2h').text.split('\n')

  cefr_estimations = []
  for this_estimation in cefr_results_breakdown:
    cefr_level, cefr_percentage, cefr_total_words = this_estimation.split(' ')
    cefr_estimations.append([cefr_level, cefr_percentage, cefr_total_words])

  df_cefr = pd.DataFrame(cefr_estimations, columns= ['cefr_level', 'percentage', 'total_words'])
  df_cefr['ratio'] = df_cefr.percentage.apply(lambda s: float(s.replace('%',''))/100.0)
  df_cefr['total_words'] = df_cefr.total_words.apply(lambda s: int(s.replace('(','').replace(')','')))

  return {'cefr_results': cefr_results, 'df_cefr': df_cefr}



# Run
baseFolder = '/Users/carlos.aguilar/Documents/chromedriver_selenium'
baseFile   = 'chromedriver'
driverFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
driver = webdriver.Chrome(driverFile)

# transcriptions here. The first time you run it, probably set to waitingTime to more that 5 seconds.
transcription = 'this does rock!'
cefr_estimation = get_CEFR_Duolingo(transcription, driver, waitingTime=5)
df_cefr = cefr_estimation['df_cefr']
overall_CEFR = cefr_estimation['cefr_results']

# when you're done
driver.close()