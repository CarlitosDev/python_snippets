from os import replace
from selenium.webdriver.support.expected_conditions import title_contains


Web scraping
------------

Selenium vs Beautiful soup

Why might you consider using Selenium? 
- Selenium is first of all a tool writing automated tests for web applications. 
- Pretty much entirely to handle the case where the content you 
want to crawl is being added to the page via JavaScript, 
rather than baked into the HTML. 


Tricks with Selenium:

driver   = webdriver.Firefox();
urlToCheck = 'http://www.theperfumeshop.com/p/1220422'
driver.get(urlToCheck)

input_element = driver.find_element_by_class_name('description')

driver.find_element_by_css_selector('div[class=''description''] > dl>dt ').text



# Revisiting this stuff - April 2020

'''
pip3 install selenium --upgrade
https://sites.google.com/chromium.org/driver/

'''
import pandas as pd
from os import path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC


baseFolder = '~/Google Drive/chromedriver'
baseFile   = 'chromedriver'
driverFile = path.expanduser(path.join(baseFolder, baseFile))
driver = webdriver.Chrome(driverFile)

city_origin = 'lond'
city_destination = 'svq'
#city_destination = 'RIX'
adults = 1
children = 0
preferdirects = 'true'
year = '20'
month = '08'

urlToCheck = f'''https://www.skyscanner.net/transport/flights/{city_origin}/{city_destination}/''' + \
f'''?adults={adults}&children={children}&adultsv2=1&childrenv2=&infants=0&cabinclass=economy&rtn=1&''' + \
f'''preferdirects={preferdirects}&outboundaltsenabled=false&inboundaltsenabled=false&''' + \
f'''oym={year}{month}&iym={year}{month}&ref=home&selectedoday=01&selectediday=01'''

driver.get(urlToCheck)


prices = [price.text for price in wait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "price")))]
flight_dates = [flightdate.text for flightdate in wait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "date")))]


flight_dates_formatted = []
flight_prices = []
for element in zip(flight_dates, prices):
    if '£' in element[1]:
        date_str = f'{element[0]}-{month}-20{year}'
        flight_dates_formatted.append(pd.to_datetime(date_str, format='%d-%m-%Y'))
        flight_prices.append(element[1].replace('£', ''))

df = pd.DataFrame({'origin':city_origin, 'destination': city_destination,
'flight_dates': flight_dates_formatted, 'prices(£)': flight_prices})

print(df)







# beautiful soup
# beautifulsoup4 

# pip install beautifulsoup4


from bs4 import BeautifulSoup
soup = BeautifulSoup("<p>Some<b>bad<i>HTML")
print(soup.prettify())

this_text = '''&lt;paragraph&gt;&lt;text&gt;&lt;input&gt;Caserta is a little city in south of Italy. It's about 800 kilometers to Milano. &lt;code a=\"CC\" c=\"There\"&gt;there&lt;/code&gt; are many beautiful &lt;code a=\"PL\" c=\"places\"&gt;place&lt;/code&gt;&lt;code a=\"PU\" c=\", \"&gt; &lt;/code&gt;&lt;code a=\"WC\" c=\"like\"&gt;as&lt;/code&gt; Reggia&lt;code a=\"RS\" c=\"\"&gt; &lt;/code&gt;, many mountains and shops.&lt;code a=\"CC\" c=\"T\"&gt;t&lt;/code&gt;his city is famous in &lt;code a=\"SP\" c=\"Italy.\"&gt;italy&lt;/code&gt;&lt;/input&gt;&lt;/text&gt;&lt;/paragraph&gt;\n'''
soup = BeautifulSoup(this_text)
print(soup.prettify())
soup.get_text()


print(soup.get_text(soup.prettify()))

'''
<paragraph>
	<text>
		<input>Caserta is a little city in south of Italy. It's about 800 kilometers to Milano. 
			<code a="CC" c="There">there</code> are many beautiful 
			<code a="PL" c="places">place</code>
			<code a="PU" c=", "></code>
			<code a="WC" c="like">as</code> Reggia
			<code a="RS" c=""></code>, many mountains and shops.
			<code a="CC" c="T">t</code>his city is famous in 
			<code a="SP" c="Italy.">italy</code>
		</input>
	</text>
</paragraph>
'''



# python3 -m pip install selenium --upgrade
# web scraping and screenshot
from selenium import webdriver

def screenshot_from_web(this_url, this_filename):
  baseFolder = '/Users/carlos.aguilar/Documents/chromedriver_selenium'
  baseFile   = 'chromedriver'
  sleep_time = 8
  driverFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
  driver = webdriver.Chrome(driverFile)
  driver.get(this_url)
  time.sleep(sleep_time)
  driver.save_screenshot(os.path.join(outputFolder, this_filename))
  driver.close()




'''

  Interact with Duolingo's CEFR estimation tool
  source ~/.bash_profile && python3 -m pip install selenium --upgrade 
'''
import pandas as pd
from os import path
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait as wait
# from selenium.webdriver.support import expected_conditions as EC


baseFolder = '/Users/carlos.aguilar/Documents/chromedriver_selenium'
baseFile   = 'chromedriver'
driverFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
driver = webdriver.Chrome(driverFile)


url_duolingo = 'https://cefr.duolingo.com/'
driver.get(url_duolingo)


inputElement = driver.find_element_by_tag_name('textarea')
inputElement.send_keys(transcription)
inputElement.send_keys(Keys.ENTER)

wait(driver, 3)

cefr_results = driver.find_element_by_class_name('RZxzk').text


cefr_results_breakdown = driver.find_element_by_class_name('_2vA2h').text.split('\n')

cefr_estimations = []
for this_estimation in cefr_results_breakdown:
  cefr_level, cefr_percentage, cefr_total_words = this_estimation.split(' ')
  cefr_estimations.append([cefr_level, cefr_percentage, cefr_total_words])

df_cefr = pd.DataFrame(cefr_estimations, columns= ['cefr_level', 'percentage', 'total_words'])
df_cefr['ratio'] = df_cefr.percentage.apply(lambda s: float(s.replace('%',''))/100.0)
df_cefr['total_words'] = df_cefr.total_words.apply(lambda s: int(s.replace('(','').replace(')','')))

driver.close()




# scrape info from local file
import os
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
# webdriver.get("file:///D:/folder/abcd.html");

baseFolder = '/Users/carlos.aguilar/Documents/chromedriver/'
baseFile   = 'chromedriver'
driverFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
driver = webdriver.Chrome(driverFile)


import carlos_utils.string_utils as stru
today_str = stru.get_today_as_string(format='%d.%m.%Y')

baseFolder = f'/Users/carlos.aguilar/Documents/EF_EVC_videos/{today_str}'
fu.makeFolder(baseFolder)
filepath = '/Users/carlos.aguilar/Documents/EF_EVC_videos/23.11.2021/private_lessons/Select Videos to change _ Adult Video Development Admin.html'
driver.get(f'file://{filepath}')


link_class_name = 'field-title'
current_row = driver.find_element_by_class_name(link_class_name)
current_row.text
current_row.


link_class_name = 'field-video'
current_row = driver.find_element_by_class_name(link_class_name)


table_xpath = '''//*[@id="result_list"]'''

this_table = driver.find_element_by_xpath(table_xpath)
this_table.size

rows = wd.findElements(By.xpath(".//*[@id='leftcontainer']/table/tbody/tr/td[1]")
table_row_count = f'''{table_xpath}/tbody/tr/td[1]'''
this_cell = driver.find_element_by_xpath(table_row_count)
not_sure_num_rows = this_cell.size['height']

import wget


for current_row in range(1, not_sure_num_rows):
  current_row


current_row = 10

title_column = 1
table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/th/a'''
this_cell = driver.find_element_by_xpath(table_accessor)
videotitle = this_cell.text

video_column = 4
table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{video_column}]/a'''
this_cell = driver.find_element_by_xpath(table_accessor)
videofile = this_cell.text.replace('Videos/', '')
videolink = this_cell.get_attribute('href')

lessonType_column = 7
table_accessor = f'''{table_xpath}/tbody/tr[{current_row}]/td[{lessonType_column}]'''
this_cell = driver.find_element_by_xpath(table_accessor)
lessonType = this_cell.text


local_file = os.path.join(baseFolder, lessonType, videofile)

wget.download(videolink, local_file)

