elasticSearchService.py


# >> USE THE BULK API
import uuid
from elasticsearch import Elasticsearch, RequestsHttpConnection #, helpers
import requests
from requests_aws4auth import AWS4Auth
import boto3

region = 'eu-west-1' 
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, \
  service, session_token=credentials.token)



endpoint = 'search-ef-lumiere-test-t7elhf63ovmmecdjpz4giwz3ie.eu-west-1.es.amazonaws.com'
index_name = 'videos_v1'
#TODO: work what this means??
document_type = '_doc'


es = Elasticsearch(
    hosts = [{'host': endpoint, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# iterator for multiple docs
actions = [
    {
        "_id" : uuid.uuid4(), # random UUID for _id
        "doc_type" : "person", # document _type
        "doc": { # the body of the document
            "name": "George Peterson",
            "sex": "male",
            "age": 34+doc,
            "years": 10+doc
        }
    }
    for doc in range(10) # use 'for' loop to insert 100 documents
]

try:
    # make the bulk call using 'actions' and get a response
    response = helpers.bulk(es, actions, index='employees', doc_type='people')
    print ("\nactions RESPONSE:", response)
except Exception as e:
    print("\nERROR:", e)