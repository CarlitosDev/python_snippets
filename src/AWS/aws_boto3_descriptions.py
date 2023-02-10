'''
	aws_boto3_descriptions.py
	From https://stackoverflow.com/questions/42809096/difference-in-boto3-between-resource-client-and-session
'''


Session:

	stores configuration information (primarily credentials and selected region)
	allows you to create service clients and resources
	boto3 creates a default session for you when needed


Client:

low-level AWS service access
generated from AWS service description
exposes botocore client to the developer
typically maps 1:1 with the AWS service API
all AWS service operations are supported by clients
snake-cased method names (e.g. ListBuckets API => list_buckets method)

	import boto3

	client = boto3.client('s3')
	response = client.list_objects_v2(Bucket='mybucket')
	for content in response['Contents']:
	    obj_dict = client.get_object(Bucket='mybucket', Key=content['Key'])
	    print(content['Key'], obj_dict['LastModified'])


** you would have to use a paginator, or implement your own loop, calling list_objects() repeatedly with a continuation marker if there were more than 1000.


Resource:

higher-level, object-oriented API
generated from resource description
uses identifiers and attributes
has actions (operations on resources)
exposes subresources and collections of AWS resources
does not provide 100% API coverage of AWS services
Here's the equivalent example using resource-level access to an S3 bucket's objects (all):

	import boto3

	s3 = boto3.resource('s3')
	bucket = s3.Bucket('mybucket')
	for obj in bucket.objects.all():
	    print(obj.key, obj.last_modified)




From session to client:
s3_client = role_session.client('s3')

From session to resource:
s3 = role_session.resource('s3')




##### Some tricks in Sagemaker
role = sagemaker.get_execution_role()
sg_session = sagemaker.Session()
region = boto3.Session().region_name
s3_client = sg_session.boto_session.client('s3')





######## Examples ########



import boto3
import utils.aws_data_utils as adu

bucket_name = 'ef-data-science-dev'
key_prefix = 'payloads'


# option 1
# To make everything easy if I run this script as
# AWS_PROFILE=efdata-dev python3
# it's all sorted



# and I can use all the methods from my helpers
bucket_contents = adu.get_buckets_contentsV2(bucket_name, key_prefix)
json_data = adu.read_json_from_s3(bucket_contents[1], bucket_name, boto3.resource('s3'))



# option 2
# But if I am on AWS lambda and need to assume the role...
role_arn_efdev = ''
role_session = adu.session_from_assumed_role(role_arn_efdev, region_name='eu-central-1')

# 2.a
# From session to client
s3_client = role_session.client('s3')
bucket_contents = adu.get_buckets_contentsV2(bucket_name, key_prefix, s3_client)
json_data = adu.read_file_from_S3(bucket_contents[1], bucket_name, s3_client)
print(json_data)

# 2.b
# From session to resource
s3_resource = role_session.resource('s3')
json_data = adu.read_json_from_s3(bucket_contents[1], bucket_name, s3_resource)




### Use the credentials directly.
import boto3
session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
		region_name='eu-west-1'
)

s3 = session.resource('s3')

bucket_name = 'ef-data-videos-amazon'
src_bucket_obj = s3.Bucket(bucket_name)

for obj in src_bucket_obj.objects.all():
	print(obj.key, obj.last_modified)