aws_secrets.py


def get_snowflake_credentials(self, secret_name = '', aws_region = ''):
    ''' 
        Fetch the credentials from AWS Secrets Manager - it requires to run under the appropiate role
    '''
    bt_session = self.get_boto3_session(aws_region)
    client = bt_session.client(service_name='secretsmanager', region_name=aws_region)

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    secretive = json.loads(get_secret_value_response['SecretString'])




