AWS_Lambda.py


Steps to get Pandas layers done:

From here:
https://medium.com/@qtangs/creating-new-aws-lambda-layer-for-python-pandas-library-348b126e9f3e

Latest installed version locally (so things are compatible):
python3 -c "import pandas; print(pandas.__version__)"


1-
mkdir aws_lambdas_for_CatBoost
echo “pandas==1.1.0\npytz==2020.1” >> requirements.txt
2-
touch get_layer_packages.sh

and make sure it contains:

	#!/bin/bash

	export PKG_DIR="python"

	rm -rf ${PKG_DIR} && mkdir -p ${PKG_DIR}

	docker run --rm -v $(pwd):/foo -w /foo lambci/lambda:build-python3.6 \
	    pip install -r requirements.txt --no-deps -t ${PKG_DIR}/lib/python3.6/site-packages/


Take note of the --no-deps flag as well, which tells pip to install only the libraries in requirements.txt and not their dependencies.

3- Layer package file

The next step is to package the zip file for the layer.
Execute the following commands from the same folder, resulting in the file my-Python36-Pandas23.zip, mimicking AWS’s aptly-named AWSLambda-Python36-SciPy1x layer.

chmod +x get_layer_packages.sh
./get_layer_packages.sh
zip -r my-Python36-Pandas110.zip .

Note that further optimizations such as remove info folder or dealing with .pyc files can be done to reduce the package size, but these are not the main interest of this post.


4- Uploading to AWS and creating a layer


	4.1- Upload the artifact to S3 
	https://s3.console.aws.amazon.com/s3/home?region=eu-west-1#
	and copy the path of your pandas_lambda.zip file in the S3 bucket
	s3://ef-data-science-dev/lambda_fncs/my-Python36-Pandas110.zip

	4.2 - Select Lambda on AWS services
	4.3 - Click the left hand panel and select layers
	4.4 - Create a new layer and set the path

5- Testing on AWS

	5.1 Create a lambda function
	https://eu-west-1.console.aws.amazon.com/lambda/home?region=eu-west-1#/create/function
	5.2 - Add a layer and select scipy and numpy.
	5.3 - Then add custom Pandas layer.
	5.4 - Write a test (skip the input)
	import json
	import pandas as pd

	def lambda_handler(event, context):
	    df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11}, \
	    {'A':'bar', 'B':'blue', 'C': 20}, \
	    {'A':'foo', 'B':'blue', 'C': 20}])
	    return {
	        'statusCode': 200,
	        'body': df.to_json(orient='records')
	    }

	5.5 Save and test




#####
# Repeat for CatBoost
# Because of the steps done above, I can skip many points
####

python3 -c "import catboost; print(catboost.__version__)"

mkdir catboost_folder
cd catboost_folder
echo catboost==0.24.1 > requirements_CatBoost.txt
rm -rf python && mkdir -p python
docker run --rm -v $(pwd):/foo -w /foo lambci/lambda:build-python3.6 \
    pip install -r requirements_CatBoost.txt --no-deps -t python/lib/python3.6/site-packages/
zip -r my-Python36-CatBoost0241.zip .


Then go to point 4



To resolve this error, set thread_count=1 in the model prediction. y_pred = model.predict(df, thread_count=1)
s3://ef-data-science-dev/lambda_fncs/my-Python36-CatBoost0241.zip



###
# To use node.js serverless
###
This guy explains it really well
https://medium.com/@qtangs/creating-new-aws-lambda-layer-for-python-pandas-library-348b126e9f3e

