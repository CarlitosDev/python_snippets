'''

Facebook's NLP
pip3 install pytext-nlp 
https://pytext.readthedocs.io/en/master/installation.html


pip3 install -U spacy
python3 -m spacy download en
python3 -m spacy download es

'''


# NLTK. Very basic sentiment analysis
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import unicodedata
from textblob import TextBlob


text = ['Mateo is sleeping', 'Carlos is fiddling around with Python', 
'Kath is a bit tired today', 'Real Madrid won Champions League',
'Rioja and beer', 'Beer or Rioja?', 'It was a great tragedy', 
'I had high expectations. I was wrong', 'Katharine is pregnant',
'Christmas is coming', 'What day is October 31?', 
'I want to buy some decorations', 'What do they say about Christmas?']
df = pd.DataFrame({'text': text})
print(df)

sid = SentimentIntensityAnalyzer()

for idx, row in df.T.iteritems():
	sentence = unicodedata.normalize('NFKD', row.text)
	ss = sid.polarity_scores(sentence)
	df.set_value(idx, 'compound', ss['compound'])
	df.set_value(idx, 'negative', ss['neg'])
	df.set_value(idx, 'neutral' , ss['neu'])
	df.set_value(idx, 'positive', ss['pos'])
	# use text 
	opinion = TextBlob(sentence)
	df.set_value(idx, 'tb_polarity', opinion.sentiment.polarity)
	df.set_value(idx, 'tb_subjectivity', opinion.sentiment.subjectivity)






#### Flair

import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
s = flair.data.Sentence(sentence)
flair_sentiment.predict(s)
total_sentiment = s.labels


