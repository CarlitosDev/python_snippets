test_beautifulsoup4.py

'''
pip3 install beautifulsoup4

'''




from bs4 import BeautifulSoup


path_to_html = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_pl_v2/21.03.2022/4e9bed89-b3fe-4a63-9c75-ca88815682ea/lesson_analysis/class_details.html'
with open(path_to_html) as fp:
  soup = BeautifulSoup(fp, "html.parser")


soup.find_all("table table-striped")

soup.find_all("a", class_="table table-striped")
a = soup.find_all(class_="table table-striped")[0]
a.find('MeetingMeta')


table_0 = soup.find_all(class_="table table-striped")[0]
for child in table_0.children:
  for idx, td in enumerate(child):
    if idx == 9:
      print(f'{idx}-{td}')
      this_td = td

type(this_td)
this_td.text

import json
lesson_features = json.loads(this_td.text.replace('\nMeetingMeta\n', ''))

