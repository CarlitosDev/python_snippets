# Confirm that we're using Python 3
import sys
assert sys.version_info.major is 3, 'Not running Python 3'


#### Resources
#
# https://www.tensorflow.org/tfx/guide/mlmd
# https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md
#
#

import os
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2



connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Empty fake database proto
store = metadata_store.MetadataStore(connection_config)



'''
Concepts
The Metadata Store uses the following data model to record and retrieve metadata from the storage backend.

ArtifactType describes a type of artifacts and their properties that are stored in the Metadata Store. These types can be registered on-the-fly with the Metadata Store in code, or they can be loaded in the store from a serialized format. Once a type is registered, its definition is available throughout the lifetime of the store.
Artifacts are specific instances of ArtifactTypes and their properties that are written to the Metadata Store.
ExecutionType describes a type of component or step in a workflow, and its runtime parameters.
Executions are records of running components or steps in an ML workflow and their runtime parameters when they ran. An Execution can be thought of as an instance of an ExecutionType (similar to how an Artifact is an instance of an ArtifactType). Every time a developer runs an ML pipeline or step, executions are recorded for each step.
Events record the relationship between Artifacts and Executions. When an Execution happens, Events record every Artifact that was used by the Execution, and every Artifact that was produced. These records allow for provenance tracking throughout a workflow. By looking at all Events, MLMD knows what Executions happened, what Artifacts were created as a result, and can recurse back from any Artifact to all of its upstream inputs.

https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#tracking-ml-workflows-with-ml-metadata
'''



# 1 - Before executions can be recorded, ArtifactTypes have to be registered.
# Create ArtifactTypes, e.g., Data and Model
# DATA TYPE
data_type = metadata_store_pb2.ArtifactType()
data_type.name = "DataSet"
data_type.properties["day"] = metadata_store_pb2.INT
data_type.properties["split"] = metadata_store_pb2.STRING
data_type_id = store.put_artifact_type(data_type)

# MODEL TYPE
model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING
model_type_id = store.put_artifact_type(model_type)

# 2 - Before executions can be recorded, ExecutionTypes have to be registered for all steps in our ML workflow.
# Create ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)


# 3 - Once types are registered, we create a DataSet Artifact.
# Declare input artifact of type DataSet
data_artifact = metadata_store_pb2.Artifact()
data_artifact.uri = 'path/to/data'
data_artifact.properties["day"].int_value = 1
data_artifact.properties["split"].string_value = 'train'
data_artifact.type_id = data_type_id
data_artifact_id = store.put_artifacts([data_artifact])


# 4 - With the DataSet Artifact created, we can create the Execution for a Trainer run
# Register the Execution of a Trainer run
trainer_run = metadata_store_pb2.Execution()
trainer_run.type_id = trainer_type_id;
trainer_run.properties["state"].string_value = "RUNNING"
run_id = store.put_executions([trainer_run])


# 5 - Declare input event and read data.
# Declare the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Submit input event to the Metadata Store
store.put_events([input_event])


# 6 - Now that the input is read, we declare the output artifact.
# Declare output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
model_artifact_id = store.put_artifacts(model_artifact)




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