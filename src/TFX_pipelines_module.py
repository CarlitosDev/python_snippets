# Confirm that we're using Python 3
import sys
assert sys.version_info.major is 3, 'Not running Python 3'

# pip3 install tfx
# https://www.tensorflow.org/tfx/guide


import tfx

from tfx import ml_metadata

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

TFX_HOME = os.path.join(os.environ['HOME'], 'tfx')
METADATA_URI = os.path.join(TFX_HOME, 'metadata/taxi/metadata.db')

conn_conf = metadata_store_pb2.ConnectionConfig()
conn_conf.sqlite.filename_uri = METADATA_URI
conn_conf.sqlite.connection_mode = 3
store = metadata_store.MetadataStore(conn_conf)

def get_uris(type_name, split):
    return [a.uri for a in store.get_artifacts_by_type(type_name) if a.properties['split'].string_value == split]

model_path = get_uris('ModelEvalPath', '')[-1] # grab last in list
result = tfma.load_eval_result(model_path)

tfma.view.render_plot(result) # shows nothing!
tfma.view.render_slicing_metrics(result) # shows nothing!



#### https://www.tensorflow.org/tfx/guide/mlmd


from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
import os


connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Empty fake database proto
store = metadata_store.MetadataStore(connection_config)



connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = os.getenv('MYSQL_SERVICE_HOST')
connection_config.mysql.port = int(os.getenv('MYSQL_SERVICE_PORT'))

connection_config.mysql.database = 'mlmetadata'
connection_config.mysql.user = 'root'
store = metadata_store.MetadataStore(connection_config)

# Get all output artifacts
store.get_artifacts()

# Get a specific artifact type

# TFX types 
# types = ['ModelExportPath', 'ExamplesPath', 'ModelBlessingPath', 'ModelPushPath', 'TransformPath', 'SchemaPath']

store.get_artifacts_by_type('ExamplesPath')