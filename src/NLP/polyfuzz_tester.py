'''
polyfuzz_tester.py

https://link.medium.com/InezL9o0lgb
https://maartengr.github.io/PolyFuzz/
source ~/.bash_profile && python3 -m pip install polyfuzz[all]
'''

from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

rapidfuzz_matcher = RapidFuzz(n_jobs=1)
model = PolyFuzz(rapidfuzz_matcher).match(from_list, to_list)
matches = model.matches



# Embeddings

from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
#https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md
from flair.embeddings import TransformerWordEmbeddings

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

bert = TransformerWordEmbeddings('bert-base-multilingual-cased')
bert_matcher = Embeddings(bert, min_similarity=0)

models = PolyFuzz(bert_matcher).match(from_list, to_list)
matches = model.matches



'''
Flair allows you to use pool word embeddings to create more powerful word embeddings. 
Below, we pool FastText and BERT to create a single embedding representation
from which we can calculate the similarity between strings
'''

from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

bert = TransformerWordEmbeddings('bert-base-multilingual-cased')
bert_matcher = Embeddings(bert, min_similarity=0)

fasttext = WordEmbeddings('en-crawl')
fasttext_matcher = Embeddings(fasttext, min_similarity=0) 

matchers = [bert_matcher, fasttext_matcher]

models = PolyFuzz(matchers).match(from_list, to_list)


'''
bert-base-multilingual-cased

(New, recommended) 12-layer, 768-hidden, 12-heads, 179M parameters.
Trained on cased text in the top 104 languages with the largest Wikipedias

en-fasttext-crawl-300d-1M.vectors.npy

wget https://flair.informatik.hu-berlin.de/resources/embeddings/token/en-fasttext-crawl-300d-1M.vectors.npy

'''



# to use word embeddings
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
fasttext_embedding = WordEmbeddings('news')
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')
embedding = Embeddings([fasttext_embedding, bert_embedding ], min_similarity=0.0)
model = pf.PolyFuzz(embedding)