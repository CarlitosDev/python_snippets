import boto3
import io
import pandas
from os import path
from subprocess import call
import gzip

def awsauth():
    call(['awsauth', 'datascience'])

def getCSV(key, bucket='beamly-marketing-science'):
    keys = get_aws_credentials()
    client = boto3.client(
        's3',
        aws_access_key_id=keys['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=keys['AWS_SECRET_ACCESS_KEY']
    )
    s3_obj = client.get_object(Key=key, Bucket=bucket)
    return pandas.read_csv(io.BytesIO(s3_obj['Body'].read()))


def put(dataframe, key, bucket='beamly-marketing-science'):
    keys = get_aws_credentials()
    client = boto3.client(
        's3',
        aws_access_key_id=keys['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=keys['AWS_SECRET_ACCESS_KEY']
    )
    body = dataframe.to_csv(index=False)
    client.put_object(Body=body, Bucket=bucket, Key=key)


def get_aws_credentials():
    with open(path.expanduser('~/.aws/env')) as f:
        env_vars_tuple = [(l.split(' ')[1].split('=')) for l in f]
        env_vars_dict = {var[0]: var[1].replace('\n', '') for var in env_vars_tuple}
        return env_vars_dict


# -----------------------------
# Carlos' functions added here:

# S3 list all keys with the prefix
def getFolderList(folderName, bucketName = 'beamly-metrics-data-stage'):
    s3Conn     = boto3.resource('s3')
    thisBucket = s3Conn.Bucket(bucketName)
    thisPrefix = folderName + '/'
    bucketList = thisBucket.objects.filter(Prefix=thisPrefix);
    folderList = [currentKey.key for currentKey in bucketList];
    return folderList

# get a list of the files within the bucket
def getBucketContentsList(bucketName = 'beamly-metrics-data-stage'):
    s3Client       = boto3.client('s3');
    bucketList     = s3Client.list_objects_v2(Bucket=bucketName);
    bucketContents = [currentKey['Key'] for currentKey in bucketList['Contents']];
    return bucketContents


def transferKeyFromBucket(currentKey, localDestination, bucketName = 'beamly-metrics-data-stage'):
    s3Conn = boto3.resource('s3')
    s3Conn.meta.client.download_file(bucketName, currentKey, localDestination)


def getKeyFromBucket(currentKey, localDestination, bucketName = 'beamly-metrics-data-stage'):
    s3Conn = boto3.resource('s3')
    obj    = s3Conn.Object(bucket_name=bucketName, key=currentKey)
    data = obj.get()["Body"].read()


# -----------------------------
# From dojo's s3/s3.py:

def compress(string):
    """
    Gzips a string using default compression level of 9, which is maximum
    :param string:
    :return: gzipped string
    """

    out = StringIO.StringIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
        f.write(string)

    return out.getvalue()


def decompress(string):
    """
    Decompresses a gzipped string
    :param string:
    :return: unzipped string; None if error
    """

    decompressed_string = gzip.GzipFile('', 'r', 0, StringIO.StringIO(string)).read()
    return decompressed_string


def read_content_as_string(filename, bucketName = 'beamly-metrics-data-stage'):
    """
    Reads from S3 and decompresses any gzip encoded content
    :param file_key:
    :return:
    """
    s3Conn     = boto3.resource('s3')
    thisBucket = s3Conn.Bucket(bucketName)
    file_key   = thisBucket.get_key(filename)
    file_content = file_key.get_contents_as_string()
    if file_key.content_encoding == 'gzip':
        return decompress(file_content)
    return file_content


# -----------------------------
# Ozan's functions added here:
def get_s3_bucket_keys(bucket_name, key_prefix):
    """ Returns a set of S3 keys for the given bucket
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    s3_keys_set = set()
    for obj in bucket.objects.filter(Prefix=key_prefix):
        s3_keys_set.add(obj.key)
    return s3_keys_set


def get_filesnames_from_s3_keys(s3_keys, regex_obj):
    """ Returns a set of file names stripped from a set of S3 keys.
    """
    source_filenames = set()
    for file in s3_keys:
        try:
            source_filenames.add(regex_obj.search(file).group())
        except AttributeError:
            # no match, then try next item in the set
            continue
    return source_filenames



#### More stuff from VMUA


def get_s3_resource_assume_role(_region_name = 'eu-central-1'):
    # Use boto3 to read the bucket contents
    sess = boto3.Session(region_name = _region_name)
    s3   = sess.resource('s3')
    return s3

def get_s3_resource(aws_access_key_id, aws_secret_access_key, _region_name = 'eu-central-1'):

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=_region_name)

    # Use boto3 to read the bucket contents
    sess = boto3.Session(region_name=_region_name)
    s3   = session.resource('s3')

    return s3

def read_xls_catalogue(xlsx_path):
    input_catalogue = pd.pandas.read_excel(xlsx_path)
    input_catalogue.product_ean.fillna(0, inplace=True)
    input_catalogue.product_ean = input_catalogue.product_ean.astype(int).astype(str)
    input_catalogue['creation_date'] = pd.to_datetime(input_catalogue['creation_date'])
    # force the integers to float
    int_colnames = input_catalogue.select_dtypes(include=['int']).columns
    input_catalogue[int_colnames] = input_catalogue[int_colnames].astype(float)
    if 'product_name' in input_catalogue.columns.tolist():
        input_catalogue.product_name = input_catalogue.product_name.str.replace('\n', '')
    return input_catalogue


def read_csv_catalogue(csv_path):
    input_catalogue = pd.read_csv(csv_path, 
        parse_dates=['creation_date'], 
        delimiter=',', 
        infer_datetime_format=True
    )

    input_catalogue.product_ean.fillna(0, inplace=True)
    input_catalogue.product_ean = input_catalogue.product_ean.astype(int).astype(str)
    #input_catalogue['creation_date'] = pd.to_datetime(input_catalogue['creation_date'])
    # force the integers to float
    int_colnames = input_catalogue.select_dtypes(include=['int']).columns
    input_catalogue[int_colnames] = input_catalogue[int_colnames].astype(float)
    if 'product_name' in input_catalogue.columns.tolist():
        input_catalogue.product_name = input_catalogue.product_name.str.replace('\n', '')
    return input_catalogue


def read_catalogue_from_s3(src_bucket_obj, src_key_prefix, product_name):
    '''

        Read all the catalogues within a certain location in S3
        Keep only map_types in (expert_model, challenger_model)

    '''
    src_full_prefix = src_key_prefix + '/' + product_name + '/'

    bucket_prefix_contents = src_bucket_obj.objects.filter(Prefix=src_full_prefix)
    # make sure no os files are there...
    src_filenames = [obj.key  for obj in bucket_prefix_contents if obj.key.find('.DS_Store') == -1]

    input_data = []
    for this_file in src_filenames:
        obj = src_bucket_obj.Object(key=this_file)
        input_data.append(read_csv_catalogue(obj.get()['Body']))

    df = pd.concat(input_data)
    # filter out the maps
    idx  = df.map_type.str.contains('expert_model|challenger_model')
    df_filtered = df[idx].copy()
    return df_filtered


def read_configuration_from_s3(src_bucket_obj, config_file, config_prefix='recommender_config'):
    '''

    Read the json configuration file from S3

    '''
    obj = src_bucket_obj.Object(key = config_prefix + '/' + config_file)
    return json.loads(obj.get()['Body'].read().decode("utf-8"))
