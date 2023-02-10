from enstop import PLSA
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


news = fetch_20newsgroups(subset='all')
data = CountVectorizer().fit_transform(news.data)

news.keys()
news['target_names']
news['target']

print(news['DESCR'])
news['filenames']

len(news['target_names'])


model = PLSA(n_components=8).fit(data)
topics = model.components_
doc_vectors = model.embedding_

data.shape
topics.shape
doc_vectors.shape


from pprint import pprint
pprint(list(news.target_names))