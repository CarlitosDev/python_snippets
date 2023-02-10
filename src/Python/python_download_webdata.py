# Read data from the web



from http import client
import urllib3
http = urllib3.PoolManager()
source = 'https://qa-et.ef-cdn.com//juno/28/74/6/v/28746/VR_12.4.3.mp4'
filename = source.split('/')[-1]
resp = http.request("GET", source)
resp.status
video_data = resp.data

video_file_test = f'/Users/carlos.aguilar/Documents/rubbish/{filename}'
with open(video_file_test, 'wb') as f:
  f.write(video_data)


import boto3
s3_client = boto3.client('s3')

import datetime
filename = source.split('/')[-1]
destination_prefix = 's3://ef-data-hyperclass/videos/adults_spaces'
key_prefix = 'videos/adults_spaces'
today = datetime.datetime.today().strftime('%d.%m.%Y')
bucket_name = 'ef-data-hyperclass'
meeting_token = 'abcd'
key_path  = f'{key_prefix}/{today}/{meeting_token}/{filename}'
destination = f'{destination_prefix}/{today}/{meeting_token}/{filename}'

response_s3 = s3_client.put_object(
    Body=video_data,
    Bucket=bucket_name,
    Key=key_path,
)

response_s3['s3_destination'] = {'bucket_name': bucket_name, 'key': key_path}

response_s3['ResponseMetadata']['HTTPStatusCode']







# read a json file
import urllib, json, requests, time, os


url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
data = json.loads(response.read())



# Download a file
url = 'https://s3.cn-north-1.amazonaws.com.cn/cn.ef.yoyo/audio/activity_26.mp3'
[fPath, fName] = os.path.split(url)
urllib.request.urlretrieve(url, fName)




# comparison for large files
# 32MB

url = 'https://eflabs-evc-record-1259390045.cos.ap-shanghai.myqcloud.com/2022/6/12/fb073b7a-cdd5-4d01-ba34-99bc95803ea9b.mp4?AWSAccessKeyId=AKIDUxLgt2d1sERZikMXJG9Xqz57cYlr5ph9&Expires=1655980525&Signature=MzfJhtCR%2FencfetMpZEMw1BPvLY%3D'
baseFolder = '/Users/carlos.aguilar/Documents/tempRubbish/videos_temp'
num_download = 0

# using requests with the stream option and different chunk_sizes
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
startTime = time.perf_counter()
chunk_size = 8192*2
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
print(f'Finished downloading {local_file}')

endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")
# Finished  in 6.15 seconds 
# second time in 18.84 seconds



# using requests with the stream option
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
startTime = time.perf_counter()
chunk_size = 8192
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
print(f'Finished downloading {local_file}')

endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")
#Finished in 18.05 seconds


# using requests with the stream option
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
startTime = time.perf_counter()
chunk_size = 8192*4
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
print(f'Finished downloading {local_file}')

endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")
# Finished in 23.15 seconds


# (2) using 
import urllib.request
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
startTime = time.perf_counter()
urllib.request.urlretrieve(url, local_file)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")
# Finished in 17.48 seconds


# (3) using wget
# It doesn't work like this...forbidden??
# wget -O [file_name] [URL]
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
wget_command = f'wget -O {local_file} {url}'
startTime = time.perf_counter()
os.system(wget_command)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")

# It doesn't work like this either...forbidden??
import subprocess
process = subprocess.Popen(
  wget_command,
  stdout = subprocess.PIPE,
  stderr = subprocess.PIPE,
  text = True,
  shell = True
)
std_out, std_err = process.communicate()
print(f'{std_out} {std_err}')



# wget_command = f'wget --recursive --level=3 -O {local_file} {url}'


# try cURL
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
curl_command = f'curl -o {local_file} {url}'
startTime = time.perf_counter()
os.system(curl_command)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")

import subprocess
process = subprocess.Popen(
  curl_command,
  stdout = subprocess.PIPE,
  stderr = subprocess.PIPE,
  text = True,
  shell = True
)
std_out, std_err = process.communicate()
print(f'{std_out} {std_err}')


# (3) try wget
# pip3 install wget
import wget
num_download +=1
local_file = os.path.join(baseFolder, f'video_{num_download}.mp4')
startTime = time.perf_counter()
wget.download(url, local_file)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")
# Finished in 13.99 seconds




retry_time = 10
from tenacity import retry, retry_if_exception_type, stop_after_delay, stop_after_attempt, wait_fixed
import urllib3.exceptions
@retry(
    retry=(
        retry_if_exception_type(urllib3.exceptions.HTTPError)
    ),
    stop=(stop_after_attempt(2)|stop_after_delay(10)),
    wait=wait_fixed(retry_time),
)
def download_file(url, local_file):
  success = False
  startTime = time.perf_counter()
  chunk_size = 8192*4
  with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
  success = True
  print(f'Finished downloading {local_file}')
  return success

startTime = time.perf_counter()
download_file(url, local_file)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")

# A more brute force retry
max_retry_time = 60
wait_retry_time = 10
from tenacity import retry, retry_if_exception_type, stop_after_delay, stop_after_attempt, wait_fixed
@retry(
    stop=(stop_after_attempt(2)|stop_after_delay(max_retry_time)),
    wait=wait_fixed(wait_retry_time),
)
def download_file(url, local_file):
  success = False
  startTime = time.perf_counter()
  chunk_size = 8192*4
  with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
  success = True
  print(f'Finished downloading {local_file}')
  return success

startTime = time.perf_counter()
download_file(url, local_file)
endTime = time.perf_counter()
runTime = endTime - startTime
print(f"Finished in {runTime:.2f} seconds")