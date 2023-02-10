'''
    Chapter 2
    https://huggingface.co/course/chapter2?fw=pt
'''

import os
import pandas as pd
import re 
import utils.file_utils as fu
import utils.utils_root as ur
import utils.content_utils as cnu
from transformers import pipeline


'''
The pipeline groups together three steps: 
1 preprocessing. tokenizers, responsible for 
    1.a,Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
    1.b,Mapping each token to an integer
    1.c,Adding additional inputs that may be useful to the model
- passing the inputs through the model
- postprocessing
'''


from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification

'''
  What this operation shows if that if we download a particular checkpoint,
  HF is able to automatically fetch the tokenizer associated with the model.
'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"# text classification
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

'''
Once we have the tokenizer, we can directly pass our sentences to it and
we’ll get back a dictionary that’s ready to feed to our model! 
The only thing left to do is to convert the list of input IDs to tensors.
Transformer models only accept tensors as input.
'''

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# inputs['input_ids'].shape >> torch.Size([2, 16]


# same idea, from the checkpoint we can download the model
# this is the architecture
model = AutoModel.from_pretrained(checkpoint)

# Feed the preprocessed inputs to our model
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# [batch_size, sequence_length, hidden_size]
# Batch size: The number of sequences processed at a time (2 in our example).
# Sequence length: The length of the numerical representation of 
# the sequence (16 in our example).
# Hidden size: The vector dimension of each model input.

'''
instead of the automodel, let's download one with a sequence classification head,
to be able to classify the sentences as positive or negative.

'''
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs_seq = model(**inputs)
print(outputs_seq.logits.shape)
# torch.Size([2, 2])
# the model head takes as input the high-dimensional vectors we saw before, 
# and outputs vectors containing two values (one per label)

import torch
predictions = torch.nn.functional.softmax(outputs_seq.logits, dim=-1)
print(predictions)

model.config.id2label



'''
  https://huggingface.co/course/chapter2/3?fw=pt
'''