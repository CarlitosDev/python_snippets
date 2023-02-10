'''
	textblob_sentiment_analysis.py
'''


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
from textblob import TextBlob

def analyse_sentiment(this_sentence):

  sentence = unicodedata.normalize('NFKD', this_sentence)
  sentiment_analysis = SentimentIntensityAnalyzer().polarity_scores(sentence)
  opinion = TextBlob(sentence)

  sentiment_analysis['tb_polarity'] = opinion.sentiment.polarity
  sentiment_analysis['tb_subjectivity'] = opinion.sentiment.subjectivity

  return sentiment_analysis