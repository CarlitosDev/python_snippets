'''

dynamic_topic_modelling.py

source ~/.bash_profile && pip3 install bertopic --upgrade


pip3 install huggingface_hub --upgrade
pip3 install -U sentence-transformers





https://towardsdatascience.com/dynamic-topic-modeling-with-bertopic-e5857e29f872
https://share.streamlit.io/sejaldua/digesting-the-digest/main/bertopic_app.py



Semisupervised mode
https://maartengr.github.io/BERTopic/tutorial/supervised/supervised.html

'''


# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


from bertopic import BERTopic
# topic_model = BERTopic(min_topic_size=70, n_gram_range=(1,3), verbose=True)
# topic_model = BERTopic(embedding_model="paraphrase-MiniLM-L3-v2", verbose=True)
embedding_model = 'all-MiniLM-L6-v2'
# topic_model = BERTopic(embedding_model="paraphrase-MiniLM-L3-v2", verbose=True)
# topic_model = BERTopic(embedding_model=embedding_model, verbose=True)



from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=model, verbose=True)



# All the summaries?
import glob
import os
import carlos_utils.file_utils as fu
baseFolder = '/Users/carlos.aguilar/Documents/EF_Content/EFxAWS/all analysed videos'
glob_pattern = os.path.join(baseFolder, '*_transcribe.json')
json_files = glob.glob(glob_pattern)[0:10]
all_transcriptions = []
for idx, this_file_path in enumerate(json_files):
  json_data = fu.readJSONFile(this_file_path)
  video_transcription = json_data['results']['transcripts'][0]['transcript']
  all_transcriptions.append(video_transcription)


# transform the {documents} to embeddings
topics, probs = topic_model.fit_transform(all_transcriptions)


topics, probs = topic_model.fit_transform([all_transcriptions[0]])

topic_model.get_topics()

# 
freq = topic_model.get_topic_info()
freq.head(10)

# topic_model.visualize_topics()
topic_nr = freq.iloc[1]["Topic"] # select a frequent topic
topic_model.get_topic(topic_nr)