!python -m pip install pyarrow==0.15
!pip install -qU awscli boto3 sagemaker



# requires PyArrow installed
train_set_fname = 'abalone_train.parquet'
test_set_fname = 'abalone_test.parquet'
train.to_parquet(train_set_fname)
test.to_parquet(test_set_fname)






sagemaker.Session().upload_data(train_set_fname, 
                                bucket=bucket, 
                                key_prefix=prefix+'/'+'train')

sagemaker.Session().upload_data(test_set_fname, 
                                bucket=bucket, 
                                key_prefix=prefix+'/'+'test')


from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(region, 'xgboost')




s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')



# I want to train on SPOT instances

sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})



hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "silent":"0",
        "objective":"reg:squarederror",
        "num_round":"50"}

instance_type = 'ml.m5.2xlarge'







# To visualise training

%matplotlib inline
from sagemaker.analytics import TrainingJobAnalytics

metric_name = 'validation:rmse'

metrics_dataframe = TrainingJobAnalytics(training_job_name=job_name, metric_names=[metric_name]).dataframe()
plt = metrics_dataframe.plot(kind='line', figsize=(12,5), x='timestamp', y='value', style='b.', legend=False)
plt.set_ylabel(metric_name);






# Ways of deploying:
# (1) from https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-deploy-model.html
xgb_predictor = xgb_model.deploy(initial_instance_count=1,
                                content_type='text/csv',
                                instance_type='ml.t2.medium'
                                )
# (2) from https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/
from sagemaker.predictor import json_serializer, csv_serializer, json_deserializer, RealTimePredictor
from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON

payload = 'M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155'
actual_rings = 10

predictor = RealTimePredictor(
    endpoint=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=csv_serializer,
    content_type=CONTENT_TYPE_CSV,
    accept=CONTENT_TYPE_JSON)

print(predictor.predict(payload))