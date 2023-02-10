# Confirm that we're using Python 3
import sys
assert sys.version_info.major is 3, 'Not running Python 3'



import tensorflow as tf
# Not fully supported on Python3 - it crashes

# Good article in here: https://medium.com/tensorflow/introducing-tensorflow-data-validation-data-understanding-validation-and-monitoring-at-scale-d38e3952c2f0
import tensorflow_data_validation as tfdv

tf.logging.set_verbosity(tf.logging.ERROR)
print('TFDV version: {}'.format(tfdv.version.__version__))



install tfx

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