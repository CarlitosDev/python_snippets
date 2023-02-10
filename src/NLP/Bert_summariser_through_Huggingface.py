# 1- Load the Huggingface model
from transformers import BertModel, BertTokenizer
model_folder = 'models'
model_to_use = 'bert-large-uncased'
huggingface_bert_model_path = os.path.join(model_folder, model_to_use)
bert_model = BertModel.from_pretrained(huggingface_bert_model_path)
bert_tokeniser = BertTokenizer.from_pretrained(huggingface_bert_model_path)

# Load the summariser
from summarizer import Summarizer
summariser_model = Summarizer(custom_model=bert_model, custom_tokenizer=bert_tokeniser)


# 0 - To download
# based on https://github.com/philschmid/serverless-bert-huggingface-aws-lambda-docker/blob/main/get_model.py
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
tokenizer.save_pretrained('.')

model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
model.save_pretrained('.')