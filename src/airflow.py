Airflow workshop
----------------
# Install the latest with extras
pip install apache-airflow[gcp_api]

It doesn't work well with Python3, so 
virtualenv -p python2.7 airflow_workshop

To begin using the virtual environment, it needs to be activated:
source airflow_workshop/bin/activate

Install the dependencies in the virtenv:
pip install apache-airflow[gcp_api]

export AIRFLOW_HOME=~/airflow

To get it started. We have to initialise the database and the webserver and
run the scheduler in another terminal:

airflow initdb

Web server:
airflow webserver -p 8080
(http://localhost:8080/admin/)

Open another terminal:
source airflow_workshop/bin/activate
airflow scheduler


Copy the dags provided in the git repo:
cp -r airflow-workshop-pydata-london-2018/dags $AIRFLOW_HOME/dags
(cp -r dags $AIRFLOW_HOME/dags)


A bit of theory:

Operators:
- sensors
- actions
- transfer

Ways of executing:
- Sequential > One task at a time
- 

- Celery executor??
- Mesos
- Kubernetes

virtualenv -p python2.7 airflow_workshop


end of Airflow workshop
----------------