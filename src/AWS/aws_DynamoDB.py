'''
  aws_DynamoDB.py

  source ~/.bash_profile && AWS_PROFILE=efdata-qa && python3

'''


import boto3

# Set up the reader
# dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
dynamodb_client = boto3.client('dynamodb', region_name='eu-west-1')
# current_table_name = 'evcLessonStatus'
# table = dynamodb.Table(current_table_name)

table_name = 'classroomAI_lesson_status'

# key schema
# https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_CreateTable.html#API_CreateTable_RequestSyntax
# Specifies the attributes that make up the primary key for a table or an index.
# AttributeName - The name of this key attribute.
# KeyType - The role that the key attribute will assume:
# HASH - partition key
# RANGE - sort key


# I only want a primary key (no need for a sort key)
key_type = [
  {'AttributeName': 'meeting_token_id',
  'KeyType': 'HASH'
  }
]

# https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_AttributeDefinition.html
# AttributeType
# The data type for the attribute, where:
# S - the attribute is of type String
# N - the attribute is of type Number
# B - the attribute is of type Binary

AttributeDefinitions=[
  {'AttributeName': 'meeting_token_id', 'AttributeType': 'S'}
]

ProvisionedThroughput={
  'ReadCapacityUnits': 1, # ReadCapacityUnits set to 1 strongly consistent reads per second
  'WriteCapacityUnits': 1  # WriteCapacityUnits set to 1 writes per second
}

table = dynamodb_client.create_table(
  TableName=table_name,
  KeySchema=key_type,
  AttributeDefinitions=AttributeDefinitions,
  ProvisionedThroughput=ProvisionedThroughput
)

waiter = dynamodb_client.get_waiter('table_exists')
waiter.wait(TableName=table_name)


item_definitions = {
  'meeting_token_id': {'AttributeType': 'S', 'DefaultValue': ''},
  'fetching_date': {'AttributeType': 'S', 'DefaultValue': ''},
  'class_id': {'AttributeType': 'S', 'DefaultValue': ''},
  'resources_downloaded': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'audio_analysed': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'transcribed': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'topic_searched': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'scenes_processed': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'sections_detected': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'feedback_analysed_NLP': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'processing_completed': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'charts_generated': {'AttributeType': 'BOOL', 'DefaultValue': False},
  'lesson_analysed': {'AttributeType': 'BOOL', 'DefaultValue': False}
}

meeting_token_id = '82fe3eac-d5b2-4ac0-a76d-b477f006ae3c'

input = {
  'meeting_token_id': meeting_token_id, 
  'fetching_date': '09.09.2022'
}

# fill the object
record = {}
for k,v in item_definitions.items():
  record[k] = {v['AttributeType']: input.get(k, v['DefaultValue'])}

import json
response = dynamodb_client.put_item(
  TableName=table_name, 
  Item=record
)

dynamodb_rs = boto3.resource('dynamodb', region_name='eu-west-1')
dynamodb_table = dynamodb_rs.Table(table_name)
response = dynamodb_table.update_item(
    Key={'meeting_token_id': meeting_token_id},
    UpdateExpression="SET resources_downloaded = :rd",
    ExpressionAttributeValues={
        ':rd': True
    },
    ReturnValues="UPDATED_NEW"
)
print(json.dumps(response, indent=4))


# TODO:
# Not sure how to make it work for a list of devices. The operator 'in' doesn't work
# list_response = []
# for thisDevice in listOfDeviceId:
#   response = table.query(KeyConditionExpression=Key('thingName').eq(thisDevice))
#   list_response.append(pd.DataFrame.from_dict(response['Items']))


# https://www.fernandomc.com/posts/ten-examples-of-getting-data-from-dynamodb-with-python-and-boto3/
# https://dynobase.dev/dynamodb-python-with-boto3/#Create-table