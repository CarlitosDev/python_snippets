'''
	AWS Kinesis Data Firehose

	You use Kinesis Data Firehose by creating a Kinesis Data Firehose delivery stream and then sending data to it.

	https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html

	For Amazon S3 destinations, streaming data is delivered to your S3 bucket. 


	
	
	The price in the US East region is $0.035 per GB of data ingested.
	
	GO:
	https://docs.aws.amazon.com/sdk-for-go/api/service/firehose/


'''

import boto3
import json
import random

kinesis = boto3.client('kinesis')

data = {}
data['heartRate'] = random.randint(150, 200)
data['rateType'] = "HIGH"

data_json = json.dumps(data)
print(data_json)
kinesis.put_record(
        StreamName="ExampleInputStream",
        Data=data_json,
        PartitionKey="partitionkey")






# More evolved example:

event_family = 'teacher'
event_topic = 'new_teacher'
partition_key = 'teacher_' + str(df_row['ProfileID'])

required_fieldnames = ['__organization', '__system', 
'__topic', '__tenant',
'__major', '__id', '__occurred']

event_producer = 'TeachersFirst'
event_id       = str(uuid.uuid4()) 
event_time     = datetime.datetime.utcnow().\
replace(tzinfo=datetime.timezone.utc).isoformat()


event_country = df_row['Country']

internal_data = {
'__organization': event_producer, 
'__system': event_family,
'__topic': event_topic, 
'__tenant': event_country,
'__major': event_version,
'__id': event_id,
'__occurred': event_time,
'tf_payload': df_row.to_dict()
}

json_data = json.dumps(internal_data, default=str)

response = kinesis_client.put_record(
    StreamName=stream_name,
    Data=json_data,
    PartitionKey=partition_key)




##### Listener

'''
  
  # Run python with the right credentials
  AWS_PROFILE=tod-prod-kinesis-producer python3


  Code adapted from here:
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kinesis.html#Kinesis.Client.get_records
  and here
  https://www.arundhaj.com/blog/getting-started-kinesis-python.html

'''

import boto3
import json
import random
import os


# Kinesis
stream_name = 'tod-prod-data-pipe'
kinesis_client = boto3.client('kinesis', region_name='eu-west-1')


# Get the stream description
stream_description = kinesis_client.describe_stream(StreamName=stream_name)

shard_id = stream_description['StreamDescription']['Shards'][0]['ShardId']

# Get an iterator
shard_iterator = kinesis_client.get_shard_iterator(StreamName=stream_name, \
  ShardId=shard_id, ShardIteratorType='LATEST')


response = kinesis_client.get_records(ShardIterator=shard_iterator['ShardIterator'], Limit=10)
print(response['Records'])

# iterate if needed
while 'NextShardIterator' in response:
    response = kinesis_client.get_records(ShardIterator=response['NextShardIterator'], Limit=10)
    print(response['Records'])