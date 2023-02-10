'''

bentoml_tutorial.py

https://docs.bentoml.org/en/latest/tutorial.html


source ~/.bash_profile
pip3 install bentoml

cd Documents/EF_repos
git clone --depth=1 git@github.com:bentoml/BentoML.git
cd bentoml/examples/quickstart/
jupyter-lab
'''



# 0 -
cd "/Volumes/GoogleDrive/My Drive/PythonSnippets/model_serving_BentoML"
# 1 - define the 'service' file
nlpqa_bentoml_service.py

# 2 - Start a dev model server to test out the service defined above
(terminal)
bentoml serve nlpqa_bentoml_service.py:svc

# 3 - test it runs. We call here the method_name
curl -X POST -H "content-type: application/text" --data "Hello Andrea,Thank you for choosing a lesson today on 'Hotels'. You did a very good job of checking into a hotel. You used the target language well and spoke clearly and confidently. Well done! I enjoyed our conversation and I look forward to seeing you on Thursday. Have a great rest of the week,Monica" http://127.0.0.1:3000/get_answers

# 4- Build Bento for deployment
Bento is the distribution format in BentoML which captures all the source code, model files, config files and dependency specifications required for running the service for production deployment. Think of it as Docker/Container designed for machine learning models.
Create the yaml file (the name has to be bentofile_{}.yaml) and run `bentoml build`

bentoml build --bentofile bentofile_nlpqa.yaml

(I don't think we need the requirements.txt file if we don't use any other libraries than the ones already defined in the yaml file.)

# 5 - build the docker container
bentoml containerize pretrained_qa_roberta_svc:latest

# 5 - Run the container
docker run -it --rm -p 3000:3000 pretrained_qa_roberta_svc:bscfbfrfhcdsrlg6 


# 6 - Have a look at the generated Dockerfile
cd /Users/carlos.aguilar/bentoml/bentos (home)
cd /Users/carlos.aguilar/bentoml/bentos/pretrained_qa_roberta_svc/bscfbfrfhcdsrlg6

# 7 - delete if needed
bentoml delete pretrained_qa_roberta_svc --yes