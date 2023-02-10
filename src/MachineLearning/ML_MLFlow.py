https://medium.com/analytics-vidhya/serving-a-model-using-mlflow-8ba5db0a26c0



Follow this article as it is the kind of idea I had for Smart Matching or on demand ML



This article uses MLFlow logger in a very welll explained manner
https://github.com/eugeneyan/papermill-mlflow
https://towardsdatascience.com/a-simpler-experimentation-workflow-with-jupyter-papermill-and-mlflow-5a84db16039



Some notes from this post:
https://towardsdatascience.com/a-true-end-to-end-ml-example-lead-scoring-f5b52e9a3c80

This one is even better:
https://towardsdatascience.com/deploying-models-to-production-with-mlflow-and-amazon-sagemaker-d21f67909198

This one is amazing but way more complex:
https://cosminsanda.com/posts/experiment-tracking-with-mlflow-inside-amazon-sagemaker/

This tutorial is very concise - LOVE IT:
https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

'''
MLflow is “An open source platform for the machine learning lifecycle.”

Easily manage and deploy machine learning models




'''

Steps:

1 - Setting up MLflow: ```bash mlflow ui```
Open http://localhost:5000/#/

You can also run a remote mlflow server/ if you’re working with a team, just make sure you specify a location for mlflow to log models to (an S3 bucket). 
See the server command below:

AWS_PROFILE=efdata-dev && mlflow server --default-artifact-root s3://ef-data-sagemaker/ELBR/mlflow --host 0.0.0.0



2 - connect to MLflow
```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LeadScoringProcessed") # creates an experiment if it doesn't exist
```

3 - Preprocessing functionality on mlflow 

To do this, we will utilize MLflow’s pyfunc model-type (more info here) which allows us to wrap up both a model and the preprocessing logic into one nice Python class. 
(more here https://www.mlflow.org/docs/latest/models.html#python-function-python-function)

4 - Logging the model to MLflow

4.1 - Set up the Anaconda env for Sagemaker:
```python
# define specific python and package versions for environment
mlflow_conda_env = {
 'name': 'mlflow-env',
 'channels': ['defaults'],
 'dependencies': ['python=3.6.2', {'pip': ['mlflow==1.6.0','scikit-learn','cloudpickle==1.3.0']}]
}
```
4.2 - we start a run within MLflow. 


4.3 - 

AWS_PROFILE=efdata-dev python3

AWS_PROFILE=efdata-dev mlflow sagemaker build-and-push-container




Concepts:
An experiment is a collection of models inside of the MLflow tracking server.



---
AWS_PROFILE=efdata-dev && jupyter-lab


-------
# this is from the lead scoring project

with mlflow.start_run(run_name="leadScoring_optimiser"):

  # if running on Spark  
  trials = SparkTrials()
  # if not on Spark
  #trials = Trials()

  # Single line bayesian optimization
  best = fmin(fn = train_hyperopt_leadScoring,
              space = param_hyperopt, 
              algo = opt_algorithm,
              trials = trials,
              max_evals = 10,
              show_progressbar=True)
   
  print(best)
  
  for key, value in space_eval(param_hyperopt, best).items():
    mlflow.log_param(key, value)
  mlflow.set_tag("project", "catboost_tester")
  mlflow.set_tag("model", "catboost_classifier")    
  mlflow.log_metric("f1", -trials.best_trial['result']['loss'])


-------
# this tutorial is a good starting point
#https://towardsdatascience.com/be-more-efficient-to-produce-ml-models-with-mlflow-c104362f377d
