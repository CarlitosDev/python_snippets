ERNIE_transformers_library.py


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en")


#From Huggingface: https://huggingface.co/sshleifer/distilbart-cnn-12-6
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")