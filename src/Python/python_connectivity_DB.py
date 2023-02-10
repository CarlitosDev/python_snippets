# Get the table names in DynamoDB
client = boto3.client('dynamodb', region_name='cn-north-1')
table_list = client.list_tables()
print(table_list['TableNames'])




# >>>> Connect to a postgres DB using SQLAlchemy
host     =  'localhost'
port     =  '5432'
user     =  'candidate'
password =  '3TqcrjTe0x1ljehu'
database =  'gocardless'

import pandas as pd
from sqlalchemy.engine import url as urlConn
from sqlalchemy import create_engine

# manually
db_conn = 'postgresql://{}:{}/{}?user={}&password={}'.format(host, port, database, user, password)
# use SQLAlchemy
urlDBConn = urlConn.URL(
		drivername='postgresql',
		username=user,
		password=password,
		host = host,
		port= port,
		database=database
		)
dbEngine = create_engine(urlDBConn)
dbConn   = dbEngine.connect()
df = pd.read_sql(sql=thisSQLQuery, con=dbConn)

dbEngine.execute(sqlQuery)

dbConn.close()



### Connect to MS SQL SERVER using SQLAlchemy
# Quick note: I had to install freetds: brew install freetds
# to make SQLALchemy work
host     =  '10.162.85.20'
user     =  'user.zz.vvv'
password =  'das@sadsaddd!'
database =  'NNNNNN'
port     =  '1433'

import pandas as pd
from sqlalchemy import create_engine

dbEngine = create_engine(f'mssql+pymssql://{user}:{password}@{host}/{database}')
dbConn   = dbEngine.connect()


tableName    = 'dbo.DimOnlineTeacher'
thisSQLQuery = f'''select top 10 * from {database}.{tableName}'''


df = pd.read_sql(sql=thisSQLQuery, con=dbConn)
df.iloc[2]


# when done
dbConn.close()


#########





# Avoid problems with special/escape characters by simply using the text helper from sqlAlchemy
from sqlalchemy import text
sqlQueryB = text(sqlQueryB)
df = rs.query2DF(sqlQueryB)




# Use boto3 to connect to Athena
region_name    = 'eu-central-1'
s3_staging_dir = 's3://aws-athena-query-results-647330586553-eu-central-1/'
credentials    = boto3.Session(region_name=region_name).get_credentials()
conn = connect(access_key=credentials.access_key,
               secret_key=credentials.secret_key,
               s3_staging_dir=s3_staging_dir,
               region_name=region_name)



# Get a the contents of an S3 bucket using resources (high-level) instead of client (low-level)
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




##################
'''
	Connect to a CockroachDB using psycopg2
	psycopg2 works at a lower level than SQLAlchemy and it needs 
	the fully qualified name and the cursor for the exection
	operations (abstracted in SQLAlchemy). 
	Also, pandas to_sql doesn't work with psycopg2

'''

import psycopg2

cr_host     = 'localhost'
cr_database = os.environ['COCKROACH_DB'].lower()
cr_user     = os.environ['COCKROACH_USER']
cr_port     = 26257

# Connect
dbConn = psycopg2.connect(
    database=cr_database,
    user=cr_user,
    sslmode='disable',
    port=cr_port,
    host=cr_host
)
# Make each statement commit immediately.
dbConn.set_session(autocommit=True)

# get the cursor
cursor = dbConn.cursor()


table = 'teacher_profile'
ext_table_name = f'{cr_database}.{table_name}'.lower()

# get the create table statement
table_name = str.lower(table)
sqlDropTable = f'DROP TABLE IF EXISTS {cr_database}.{table_name};'
cursor.execute(sqlDropTable)


##################
'''
	Connect to a CockroachDB using SQLAlchemy
	(insecure mode)

	pip3 install sqlalchemy cockroachdb

'''

from sqlalchemy.engine import url as urlConn
from sqlalchemy import create_engine

urlDBConn = urlConn.URL(
    drivername='cockroachdb',
    database=cr_database,
    username=cr_user,
    port=cr_port,
    host=cr_host
)

connect_args = {'sslmode': 'disable'}


dbEngine = create_engine(urlDBConn, connect_args=connect_args,echo=True)
dbConn = dbEngine.connect()




# port forwarding
from sshtunnel import SSHTunnelForwarder
import psycopg2
def SB_read_PROD_postgresql_ssh(sqlQuery):
    '''
        Query SB PostgreSQL DB using port forwarding
    '''
    df = pd.DataFrame()

    sql_hostname = os.environ['SB_PROD_POSTGRESQL_HOST']
    sql_username = os.environ['SB_PROD_POSTGRESQL_USER']
    sql_password = os.environ['SB_PROD_POSTGRESQL_PASS']
    sql_main_database = 'yoyodev'
    sql_port = 5432

    with SSHTunnelForwarder(
        (os.environ['SB_PROD_SSH_HOST']),
        ssh_username=os.environ['SB_PROD_SSH_USER'],
        ssh_pkey="/Users/carlos.aguilar/.ssh/sb-prod-jumpbox.pub",
        ssh_private_key_password="",
        allow_agent=True,
        remote_bind_address=(sql_hostname, sql_port)
    ) as tunnel:
        print('Tunnelled. Ready to connect')
        conn = psycopg2.connect(host="127.0.0.1",
                                port=tunnel.local_bind_port,
                                database=sql_main_database,
                                user=sql_username,
                                password=sql_password)
        try:
            df = pd.read_sql(sql=sqlQuery, con=conn)
        finally:
            conn.close()
    return df