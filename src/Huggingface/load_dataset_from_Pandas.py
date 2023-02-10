'''
  load_dataset_from_Pandas.py

'''



import pandas as pd
from datasets import Dataset, DatasetDict
import carlos_utils.file_utils as fu


path = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/Education First/writing_corrections/corrections.pickle'
writing_corrections = fu.readPickleFile(path)

original_text = []
corrected_text = []
for corrections in writing_corrections.values():
  if corrections['corrected']:
    original_text.append(corrections['original_text'])
    corrected_text.append(corrections['corrected_text'])


df = pd.DataFrame({'original_text': original_text, 'corrected_text': corrected_text})

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)




# Into huggingface datasets.
dataset = DatasetDict()
dataset['train'] = df_train
dataset['test'] = df_test

print(dataset)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


from random import randrange        
sample = dataset['train'].loc[randrange(len(dataset["train"]))]
print(f"original_text: \n{sample['original_text']}\n---------------")
print(f"corrected_text: \n{sample['corrected_text']}\n---------------")