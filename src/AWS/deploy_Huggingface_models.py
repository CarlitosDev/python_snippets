'''
deploy_Huggingface_models.py

source ~/.bash_profile && AWS_PROFILE=efdata-qa
pip3 install sagemaker
python3


This should be covered in here
https://huggingface.co/docs/sagemaker/inference
'''

# 
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import DataSerializer

region = 'eu-west-1'
iam_client = boto3.client('iam', region)
role_name = 'ef-qa-classroomAI'
role = iam_client.get_role(RoleName=role_name)

# session = boto3.session.Session(region_name=region)
# sagemaker_client = session.client('sagemaker')


import carlos_utils.file_utils as fu
fu.printJSON(role)

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'openai/whisper-medium',
	'HF_TASK':'automatic-speech-recognition'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.17.0',
	pytorch_version='1.10.2',
	py_version='py38',
	env=hub,
	role=role['Role']['Arn']
)

# CPU
ec2_instance_type = 'ml.m5.xlarge'
# GPU
ec2_instance_type = 'ml.g4dn.xlarge'

# using x-audio to support multiple audio formats
audio_serializer = DataSerializer(content_type='audio/x-audio')


# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type=ec2_instance_type,
  serializer=audio_serializer, # serializer for our audio data.
)

print(predictor.endpoint_name)

'''
From https://www.philschmid.de/automatic-speech-recognition-sagemaker

We will use 2 different methods to send requests to the endpoint:
	a. Provide a audio file via path to the predictor
	b. Provide binary audio data object to the predictor
'''

# This doesn't work...can't open file in S3
# response = predictor.predict(
# 	data='s3://ef-data-hyperclass/videos/adults_spaces/08.09.2022/1385399d-19e6-4cad-802c-e08fa28fbd49/evc_API/1385399d-19e6-4cad-802c-e08fa28fbd49b.flac'
# )

'''
botocore.exceptions.SSLError: SSL validation failed for https://runtime.sagemaker.eu-west-1.amazonaws.com/endpoints/huggingface-pytorch-inference-2022-10-18-12-29-02-812/invocations EOF occurred in violation of protocol (_ssl.c:2396)
'''
local_path_to_audiofile = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_Hyperclass_videos/adults_spaces/09.09.2022/653d219c-d874-4ad1-9e3c-8808e1b19a9c/evc_API/653d219c-d874-4ad1-9e3c-8808e1b19a9cb.flac'
response = predictor.predict(data=local_path_to_audiofile)



'''
botocore.exceptions.SSLError: 
SSL validation failed for 
https://runtime.sagemaker.eu-west-1.amazonaws.com/endpoints/huggingface-pytorch-inference-2022-10-18-12-29-02-812/invocations 
EOF occurred in violation of protocol (_ssl.c:2396)
'''
with open(local_path_to_audiofile, "rb") as data_file:
  audio_data = data_file.read()
  res = predictor.predict(data=audio_data)
  print(res)


# with open(local_path_to_audiofile, "rb") as data_file:
#   audio_data = data_file.read()

# res = predictor.predict(data=audio_data[0:1000000])
# print(res)


local_path_to_sample ='/Users/carlos.aguilar/Documents/tempRubbish/sample1.flac'
response = predictor.predict(data=local_path_to_sample)

predictor.delete_model()
predictor.delete_endpoint()
