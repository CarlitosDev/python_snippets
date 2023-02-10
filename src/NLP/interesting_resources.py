interesting_resources.py

https://www.theseattledataguy.com/how-to-process-pdfs-and-documents-with-aws-comprehend-and-gcp/#page-content
>>In the article SDG shows how to use AWS Comprehend and TextBlob



Some notes:

#Tokenization is the processing of segmenting text into sentences or words. In the process, we throw away punctuation and extra symbols too.

import nltk
sentence = "My name is microSD, so I am a SD card"
tokens = nltk.word_tokenize(sentence)
print(tokens)

#Stop Words Removal	


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
filtered_tokens = [w for w in tokens if w not in stop_words]
print(filtered_tokens)

#Stemming is the process of reducing words into their root form. 

terms_to_stem = ['cook', 'cooking', 'cooked', 'cooks']
snowball_stemmer = nltk.stem.SnowballStemmer('english')

for iTerm in terms_to_stem:
	print(snowball_stemmer.stem(iTerm)


# Word Embeddings
#
# Representing words as numbers, in such a way that words with similar meaning have
# a similar representation. Modern-day word embeddings represent individual words
# as real-valued vectors in a predefined vector space.

# A common method for creating word embeddings is called GloVe, which stands for “Global Vectors”. 
# GloVe captures global statistics and local statistics of a text corpus in order to create word vectors.


# Term Frequency-Inverse Document Frequency
#
# TF-IDF is a weighting factor often used in applications such as information
# retrieval and text mining. TF-IDF uses statistics to measure
# how important a word is to a particular document.