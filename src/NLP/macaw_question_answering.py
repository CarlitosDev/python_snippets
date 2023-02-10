# MACAW_QuestionAnswering
#https://github.com/allenai/macaw
# paper
# https://medium.com/ai2-blog/general-purpose-question-answering-with-macaw-84cd7e3af0f7
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
input_string = "$answer$ ; $mcoptions$ ; $question$ = What is the color of a cloudy sky?"
input_ids = tokenizer.encode(input_string, return_tensors="pt")
output = model.generate(input_ids, max_length=200)
tokenizer.batch_decode(output, skip_special_tokens=True)


# I love it..."stay"
input_string = "$answer$ ; $mcoptions$ ; $question$ = Should I leave or should I go?"
input_ids = tokenizer.encode(input_string, return_tensors="pt")
output = model.generate(input_ids, max_length=200)
tokenizer.batch_decode(output, skip_special_tokens=True)