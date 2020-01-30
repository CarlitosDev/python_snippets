# Read data from the web


# read a json file
import urllib, json

url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
data = json.loads(response.read())



# Download a file
import urllib.request
import os
url = 'https://s3.cn-north-1.amazonaws.com.cn/cn.ef.yoyo/audio/activity_26.mp3'
[fPath, fName] = os.path.split(url)
urllib.request.urlretrieve(url, fName)
