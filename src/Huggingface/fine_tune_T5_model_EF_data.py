'''
  fine_tune_T5_model.py for Abstractive Summarization (text2text-generation task).

  T5 and FLAN-T5:
  FLAN-T5 released with the Scaling Instruction-Finetuned Language Models paper is an enhanced version of T5 that has been finetuned in a mixture of tasks. 

  [FLAN-T5: https://github.com/google-research/t5x]

  Recommended EC2 instance: 
  g4dn.xlarge AWS EC2 Instance including a NVIDIA T4.




  From: 
  https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
  
  pip3 install transformers datasets rouge-score nltk tensorboard py7zr evaluate --upgrade

  See this one:
  https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887


  NLP transfer learning: T5 models reframe any NLP task such that both the input and the output are text sequences. 
  The same T5 model can be used for any NLP task, without any aftermarket changes to the architecture.
  The task to be performed can be specified via a simple prefix (again a text sequence) prepended to the input.

  # Training tricks:
  Note that you can offset the effect of a small batch size by increasing the gradient_accumulation_steps. 
  The effective batch size is roughly equal to train_batch_size * gradient_accumulation_steps.

'''



import pandas as pd
from datasets import concatenate_datasets, DatasetDict, Dataset, load_dataset
import carlos_utils.file_utils as fu
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

path = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/Education First/writing_corrections/corrections.pickle'
writing_corrections = fu.readPickleFile(path)

original_text = []
corrected_text = []
for corrections in writing_corrections.values():
  if corrections['corrected']:
    original_text.append(corrections['original_text'])
    corrected_text.append(corrections['corrected_text'])


df = pd.DataFrame({'original_text': original_text, 'corrected_text': corrected_text})
df = df[0:250].copy()

# Load the tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_id="google/flan-t5-base"
# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get largest padding
df['original_text_len'] = df['original_text'].apply(len)
df['corrected_text_len'] = df['corrected_text'].apply(len)
df['max_text_len'] = df[['original_text_len', 'corrected_text_len']].max(axis=1)
df.sort_values('max_text_len', inplace=True, ascending=False)

tokenized_inputs = df['original_text'][0:50].map(lambda x: tokenizer(x, truncation=True))
tokenized_inputs_len = tokenized_inputs.apply(lambda x: len(x['input_ids']))
max_source_length = max(tokenized_inputs_len)
print(f"Max source length: {max_source_length}")

tokenized_targets = df['corrected_text'][0:50].map(lambda x: tokenizer(x, truncation=True))
tokenized_targets_len = tokenized_targets.apply(lambda x: len(x['input_ids']))
max_target_length = max(tokenized_targets_len)
print(f"Max target length: {max_target_length}")



# Split
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)


# Into huggingface datasets.
dataset = DatasetDict()
dataset['train'] = Dataset.from_pandas(df_train)
dataset['test'] = Dataset.from_pandas(df_test)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


from random import randrange
sample = dataset['train'][randrange(len(dataset["train"]))]
print(f"original_text: \n{sample['original_text']}\n---------------")
print(f"corrected_text: \n{sample['corrected_text']}\n---------------")



# task_type='summarize'
def preprocess_function(sample, padding="max_length", task_type='grammar', debug_output=False):
    # add prefix to the input for t5
    inputs = [f"{task_type}: " + item for item in sample["original_text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["corrected_text"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    if debug_output:
      model_inputs["text_inputs"] = inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["original_text", "corrected_text"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Inspect if this makes sense
# a = dataset['train'].map(preprocess_function, batched=True)
# a['text_inputs'][10]

train = tokenized_dataset['train']
type(train.data)

import os
repository_id = os.path.join(os.environ['TEMP_DATA_DIR'], \
  model_id.split('/')[1], 'writing_corrections')
fu.makeFolder(repository_id)

tokenized_dataset.save_to_disk(os.path.join(repository_id, 'tokenized_dataset'))

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)



import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result



from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)



from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id


# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
tokenizer.save_pretrained(repository_id)
trainer.evaluate()
