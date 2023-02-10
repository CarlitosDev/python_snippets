# Read csv and xlsx data from S3
import pandas as pd
import boto3
import os
# Tip: to save the credentials, EXPORT the variable and source it
s3 = get_s3_resource(
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'],
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
)

src_bucket_name = 'beamly-data-vmua'
if read_csv:
    src_key_prefix  = 'recommender_data/csv/'
else:
    src_key_prefix  = 'recommender_data/xls/'


src_bucket_obj  = s3.Bucket(src_bucket_name)
bucket_prefix_contents = src_bucket_obj.objects.filter(Prefix=src_key_prefix)

# full path to filenames
src_filenames = [obj.key  for obj in bucket_prefix_contents]


product_names = ['foundation', 'concealer', 'blusher',
                    'mascara', 'eyeshadow', 'eyeliner', 'eyebrows', 'lipstick']

for idx, product_name in enumerate(product_names, 0):
    this_list = [this_file for this_file in src_filenames if this_file.lower().find(product_name) != -1]
    # here iterate through the keys
    obj = src_bucket_obj.Object(key=this_list[0])
    if read_csv:
        df = pd.read_csv(obj.get()['Body'])
    else:
        df = read_catalogue(obj.get()['Body'])
    
    df.head()




#####
# Explicitily instanciate the client with the keys
client = boto3.client('s3',
    aws_access_key_id=keys['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=keys['AWS_SECRET_ACCESS_KEY'])

# Explicitily instanciate the session with the keys
session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                  aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                  aws_session_token=response['Credentials']['SessionToken'])

####




# get a list of the files within the bucket
def getBucketContentsList(bucketName = 'beamly-metrics-data-stage'):
    s3Client       = boto3.client('s3');
    bucketList     = s3Client.list_objects_v2(Bucket=bucketName);
    bucketContents = [currentKey['Key'] for currentKey in bucketList['Contents']];
    return bucketContents




# This one works
s3Client       = boto3.client('s3');
bucket_name = "beamly-data-qubole-prod"
key_prefix  = "data/masterdataset/adform parquet/tracking points/"
bucketList     = s3Client.list_objects_v2(Bucket=bucket_name, Prefix=key_prefix);
bucketContents = [currentKey['Key'] for currentKey in bucketList['Contents']];




%spark.pyspark
import boto3
import boto3.session
session = boto3.session.Session(region_name='eu-central-1')
s3client = session.client('s3', config=boto3.session.Config(signature_version='s3v4'),aws_access_key_id=keys['AWS_ACCESS_KEY_ID'], aws_secret_access_key=keys['AWS_SECRET_ACCESS_KEY'])
bucket_name = "beamly-data-qubole-prod"
key_prefix  = "data/masterdataset/adform parquet/tracking points/"
bucketList     = s3Client.list_objects_v2(Bucket=bucket_name, Prefix=key_prefix);
bucketContents = [currentKey['Key'] for currentKey in bucketList['Contents']];
print(bucketContents[32])




# This one works in Qubole
%spark.pyspark
import boto3
sess = boto3.Session(region_name='eu-central-1')
s3 = sess.resource('s3')

b = s3.Bucket("beamly-data-qubole-prod")
list(b.objects.all())


# S3 Object (bucket_name and key are identifiers)
%spark.pyspark
import boto3
sess = boto3.Session(region_name='eu-central-1')
s3 = sess.resource('s3')
bucket_name = "beamly-data-qubole-prod"
bucketObj = s3.Bucket(bucket_name)

# filter with the prefix
key_prefix  = "data/masterdataset/adform parquet/tracking points/"
bucketPrefixContents = bucketObj.objects.filter(Prefix=key_prefix) 
bucketContents = ['s3://' + bucket_name + '/' + obj.key for obj in bucketPrefixContents]




for obj in bucketContents:
	print('{0}'.format(obj.key))


# S3 list all keys with the prefix 'photos/'
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    for obj in bucket.objects.filter(Prefix='photos/'):
        print('{0}:{1}'.format(bucket.name, obj.key))





###### To overcome the limitation of 1000 tokens

def get_all_s3_objects(s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')

for file in get_all_s3_objects(boto3.client('s3'), Bucket=bucket, Prefix=prefix):
    print(file['size'])