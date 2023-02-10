'''
get infgo about the instance running the notebook
'''
def get_instance_info():    
    log_path = '/opt/ml/metadata/resource-metadata.json'
    with open(log_path, 'r') as logs:
        _logs = json.load(logs)
    return _logs

nb_logs = get_instance_info()
for k,v in nb_logs.items():
    print(f'{k}:{v}')


'''

	aws_sagemaker.py


https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/US-census_population_segmentation_PCA_Kmeans/sagemaker-countycensusclustering.ipynb

'''


from sagemaker import get_execution_role
role = get_execution_role()




import boto3



# Do PCA on a large dataset

from sagemaker import PCA

bucket = 'lolo_bucket'

num_components=33
pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path='s3://'+ bucket +'/counties/',
             num_components=num_components)


pca_SM.fit(pca_SM.record_set(train_data))



'''
From here:
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb
'''

# Upload the pre-trained model to S3
fObj = open("model.tar.gz", 'rb')
key= os.path.join(prefix, model_file_name, 'model.tar.gz')
boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fObj)


# Create a model object from a tar file in S3:
from time import gmtime, strftime

model_name = model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model_url = 'https://s3-{}.amazonaws.com/{}/{}'.format(region,bucket,key)
sm_client = boto3.client('sagemaker')

print (model_url)

primary_container = {
    'Image': container,
    'ModelDataUrl': model_url,
}

create_model_response2 = sm_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = primary_container)

print(create_model_response2['ModelArn'])