some_day_with_BERT.py


from summarizer.bert_parent import BertParent
bertObj = BertParent('bert-large-uncased')
bertObj.model

tokenized_input = bertObj.tokenizer(user_input)
from torch import tensor as ttensor
ttensor(tokenized_input).unsqueeze(0)  # Batch size 1
bertObj.model()

input_ids = tokenized_input['input_ids']
outputs = bertObj.model(input_ids)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

