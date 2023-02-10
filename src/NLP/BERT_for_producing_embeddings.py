

'''

Tutorial here:
https://medium.com/analytics-vidhya/recommendation-system-using-bert-embeddings-1d8de5fc3c56

Data from here: https://www.kaggle.com/jyotmakadiya/top-trending-videos-youtube-2021?select=GB_videos_data.csv


It uses tensorflow HUB to fetch the models.


Requirements:
source ~/.bash_profile && python3 -m pip install tensorflow_hub --upgrade

python3 -m pip install tensorflow-text --upgrade


My summary:
  - Get the titles of several YouTube videos (GB)
  - Basic cleanup on the titles
  - Leverage Tensorflow Hub to download pretrained BERT preprocessor and encoder
  - Apply it to the dataset
  - Then type queries in the prompt and get a match


'''

import os
import pandas as pd
import numpy as np
import re

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


from sklearn.metrics.pairwise import cosine_similarity

filepath = '/Users/carlos.aguilar/Documents/Kaggle/TopTrendingVideos YouTube2021/GB_videos_data.csv'

fcn_clean_text = lambda x: re.sub('[^A-Za-z0-9]+', ' ', str(x).lower())

df_yt = pd.read_csv(filepath)
df_yt = df_yt.drop_duplicates(subset = ['title'])
df_yt = df_yt[['title', 'description']]
df_yt.columns = ['Title', 'Description']
#df_yt['cleaned_title'] = df_yt['Title'].apply(lambda x: x.lower())
df_yt['cleaned_title'] = df_yt['Title'].apply(fcn_clean_text)

df_yt['cleaned_description'] = df_yt['Description'].apply(fcn_clean_text)


preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1", trainable=True)

def get_bert_embeddings(this_text, preprocessor, encoder):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encoder_inputs = preprocessor(text_input)
  outputs = encoder(encoder_inputs)
  embedding_model = tf.keras.Model(text_input, outputs['pooled_output'])
  sentences = tf.constant([this_text])
  return embedding_model(sentences)


# Encodings is a vector of Nx128
df_yt['encodings'] = df_yt['cleaned_title'].apply(lambda x: get_bert_embeddings(x, preprocessor, encoder))
df_yt['encodings_desc'] =df_yt['cleaned_description'].apply(lambda x: get_bert_embeddings(x, preprocessor, encoder))

# This method below captures a query from the prompt
def preprocess_text():
  text = input()
  text = text.lower()
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  return text
  
# capture the query
query_text = preprocess_text()

query_encoding = get_bert_embeddings(query_text, preprocessor, encoder)

df_yt['similarity_score'] = df_yt['encodings'].apply(lambda x: cosine_similarity(x, query_encoding)[0][0])
df_results = df_yt.sort_values(by=['similarity_score'], ascending=False)
df_results.head(10)