EC2_instance_setup.py


GPU-DL train only
g4dn.4xlarge


In the IAM instance profile select "ec2-delete"



sudo apt update && sudo apt install python3-pip jupyter-notebook
sudo apt update && sudo apt install ffmpeg jupyter-notebook
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -qq awscliv2.zip
sudo ./aws/install

source activate pytorch

pip3 install --upgrade ipywidgets boto3 
pip3 install scikit-learn transformers datasets rouge-score nltk tensorboard py7zr evaluate --upgrade
pip3 install markupsafe==2.0.1
pip3 install pandas torch
jupyter nbextension enable --py widgetsnbextension




mkdir -p fineTuneT5
cd  fineTuneT5
aws s3 cp s3://ef-data-writing-corrections/fineTrainT5/code.zip .
unzip -qq code.zip




jupyter notebook --allow-root --port 8888 --ip 0.0.0.0


http://3.252.155.239:8888/?token=3b761c6c2d9f4c859f935e7b7b572b33462201f1d73472bd
http://3.252.155.239:8888/?token=aa2b795d30608d8700747885665c42be5ae162f7fc4b97e0



pip3 install torch --no-cache-dir


## When using a DLAMI
# https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html
source activate pytorch
# jupyter comes installed
pip3 install scikit-learn transformers datasets rouge-score nltk py7zr evaluate --upgrade
jupyter notebook --allow-root --port 8888 --ip 0.0.0.0


http://127.0.0.1:8888/?token=26060369555af16c7c98f1104a5865b05870175fa2f00efb