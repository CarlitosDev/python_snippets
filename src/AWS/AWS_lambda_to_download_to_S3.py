'''
AWS_lambda_to_download_to_S3.py

Remember to increase the timeout to at least 10 mins and the memory to at least 512MB (as the videos live in memory)

'''


import json
import subprocess
import os
import urllib3
import boto3
from datetime import datetime

# TODO: Increase lambda's RAM to allocate for these videos. 512 should be more than enough
def download_url_content(url):
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    return resp



def lambda_handler(event, context):
    
    meeting_token_id = event.get('meeting_token_id', None)
    
    source = event.get('url_source', None)
    filename = source.split('/')[-1]
    resp = download_url_content(source)
    
    if resp.status == 200:
        response_s3 = write_to_S3(resp.data, filename, meeting_token_id)
        
    return {
    'statusCode': 200,
    'body': json.dumps(response_s3)
    }
    

def write_to_S3(input_data: bytes, filename: str, meeting_token_id: str, bucket_name: str = 'ef-data-hyperclass', key_prefix:str = 'videos/adults_spaces'):
    
    s3_client = boto3.client('s3')
    today = datetime.today().strftime('%d.%m.%Y')
    key_path  = f'{key_prefix}/{today}/{meeting_token_id}/evc_API/{filename}'

    response_s3 = s3_client.put_object(
        Body=input_data,
        Bucket=bucket_name,
        Key=key_path,
    )
    response_s3['s3_destination'] = {'bucket_name': bucket_name, 'key': key_path}
    return response_s3