'''


  STEPS to get Selenium up and running:

  1 - Get Chromium from here https://chromedriver.chromium.org/
    1.a - Mind that the Chromium version must match the installed Chrome version.
    1.b - Execute this on the Chromium file:
    cd '/Volumes/GoogleDrive/My Drive/chromedriver/'
    spctl --add --label 'Approved' chromedriver
    xattr -d com.apple.quarantine "/Users/carlos.aguilar/Documents/chromedriver/chromedriver"
  2 - PIP install Selenium
  3 - Python code below

  Updates:
  23.11.2021 - First attempt


  Runner:
  cd '/Users/carlos.aguilar/Documents/EF_repos/data_science_utilities/src';
  
  cd '/Users/carlos.aguilar/Documents/carlosDev/data_science_utilities/src';
  source ~/.bash_profile && python3 ./EVC_videos/parse_app_download_videos.py


# Sync the local downloads in the bucket 'ef-data-evc-videos'
source ~/.bash_profile && AWS_PROFILE=efdata-qa && aws s3 sync "/Users/carlos.aguilar/Documents/EF_EVC_videos/23.11.2021/Lesson Recording Adults Private Lesson/" "s3://ef-data-evc-videos/"



'''

import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.select import Select
import carlos_utils.string_utils as stru
import carlos_utils.file_utils as fu
import wget
import time

chromiumFolder = '/Users/carlos.aguilar/Documents/chromedriver/'
baseFile   = 'chromedriver'
driverFile = os.path.expanduser(os.path.join(chromiumFolder, baseFile))
driver = webdriver.Chrome(driverFile)




today_str = stru.get_today_as_string(format='%d.%m.%Y')

baseFolder = f'/Users/carlos.aguilar/Documents/EF_EVC_videos/{today_str}'
fu.makeFolder(baseFolder)

url = 'https://teachonlinevideodevelopment.azurewebsites.net/admin/data/view/ManageVideo/videomodel/'
driver.get(url)

username_xpath = '//*[@id="id_username"]'
username_element = driver.find_element_by_xpath(username_xpath)
username_element.send_keys('carlos.aguilar@ef.com')
time.sleep(2)

password_xpath = '//*[@id="id_password"]'
password_element = driver.find_element_by_xpath(password_xpath)
carlos_password = os.environ['teachonline_video_app_password']
password_element.send_keys(carlos_password)
time.sleep(2)



login_xpath = '//*[@id="login-form"]/div[3]/input'
login_element = driver.find_element_by_xpath(login_xpath)
login_element.click()
time.sleep(10)

table_xpath = '''//*[@id="result_list"]'''
this_table = driver.find_element_by_xpath(table_xpath)
table_row_count = f'''{table_xpath}/tbody/tr/td[1]'''
this_cell = driver.find_element_by_xpath(table_row_count)
not_sure_num_rows = this_cell.size['height']



select_element = Select(driver.find_element_by_xpath('//*[@id="changelist-filter"]/div[3]/div/select'))
select_element.select_by_index(1)
time.sleep(10)

#
# not_sure_num_rows
max_daily_videos = 15


def save_videos():
  # for current_row in range(1, not_sure_num_rows):
  for current_row in range(1, max_daily_videos):  
    title_column = 1
    table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/th/a'''
    this_cell = driver.find_element_by_xpath(table_accessor)
    videotitle = this_cell.text

    id_column = 2
    table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{id_column}]'''
    this_cell = driver.find_element_by_xpath(table_accessor)
    videoid = this_cell.text

    video_column = 4
    table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{video_column}]/a'''
    this_cell = driver.find_element_by_xpath(table_accessor)
    videofile = this_cell.text.replace('Videos/', '')
    videolink = this_cell.get_attribute('href')

    lessonType_column = 7
    table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{lessonType_column}]'''
    this_cell = driver.find_element_by_xpath(table_accessor)
    lessonType = this_cell.text



    lessonType_column = 8
    table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{lessonType_column}]'''
    this_cell = driver.find_element_by_xpath(table_accessor)
    lessonCreated = this_cell.text

    fu.makeFolder(os.path.join(baseFolder, lessonType))
    local_file = os.path.join(baseFolder, lessonType, videofile)
    
    if not os.path.exists(local_file):
      print(f'Donwloading {local_file}...')
      wget.download(videolink, local_file)

      video_info = {'videoTitle':videotitle,
      'videoFile': videofile,
      'videoId': videoid,
      'lessonType': lessonType,
      'lessonCreated': lessonCreated
      }
      json_path = os.path.join(baseFolder, lessonType, videofile.replace('mp4', 'json'))
      fu.writeJSONFile(video_info, json_path)



save_videos()

select_element = Select(driver.find_element_by_xpath('//*[@id="changelist-filter"]/div[3]/div/select'))
select_element.select_by_index(2)
time.sleep(10)

save_videos()


select_element.select_by_index(3)
time.sleep(10)
save_videos()