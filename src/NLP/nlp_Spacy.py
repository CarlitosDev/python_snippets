'''
source ~/.bash_profile && python3 -m pip install -U spacy en-core-web-trf en-core-web-sm en-core-web-lg
python3 -m spacy download en
python3 -m spacy download es


https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.1.0/en_core_web_lg-3.1.0-py3-none-any.whl


Good tutorial: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
'''

'''
https://github.com/explosion/spacy-course
'''



# Ask spacy to explain the tags
spacy.explain("PROPN")


# > Bring models from other libraries
# Get fast-text
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.la.300.vec.gz
python -m spacy init-model en /tmp/la_vectors_wiki_lg --vectors-loc cc.la.300.vec.gz
nlp_latin = spacy.load("/tmp/la_vectors_wiki_lg")
doc1 = nlp_latin("Caecilius est in horto")
doc2 = nlp_latin("servus est in atrio")
doc1.similarity(doc2)




from spacy import displacy
displacy.serve(doc, style="dep")


'''
To download and use a model
source ~/.bash_profile && python3 -m spacy download en_core_web_lg
'''
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
doc = nlp(u"This is a sentence.")



# work out where the models are stored
import spacy
model = spacy.load("en_core_web_sm")
model._path
# PosixPath('/usr/local/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')

'''
  Basic preprocessing
'''


import spacy
nlp = spacy.load("en_core_web_sm")

transcription = '''and I think that there is one thing that I do absolutely love away
from the Children and that is that my day to day grind, what I do for a living and how I do
it being both a family person with the kids and taking care of the house and working, doing 
my job professionally is that my day can now be broken up. It wasn't like this when when I
was in my twenties. My day. I don't have to work 9 to 5 Monday through Friday. I can work parts 
of the day because of technology has given me access to different times. I can video conference 
different times that I can easily get on a social network or uh, like an email platform and
get the research done I need for my work. I can access the programs that we have at work
at different times in the middle of the night. You know, when I don't need to be doing 
stuff at home for the kids. This is something that actually is really great about technology'''

current_doc = nlp(transcription)


# Tokenization
tokenised_text = [word.text for word in current_doc]

# Remove stopwords
#all_stopwords = set([*nlp.Defaults.stop_words] + ['.',',','\n'])
all_stopwords = nlp.Defaults.stop_words
text_no_stopwords = [word.text for word in current_doc if not word in all_stopwords]




# Lemmatization is an organized & step-by-step procedure of obtaining
# the root form of the word. It makes use of vocabulary 
# (dictionary importance of words) 
# and morphological analysis (word structure and grammar relations).
lemmatised_text = [word.lemma_ for word in current_doc if not word in all_stopwords]

# Get the sentences
sentences = [this_sentence for this_sentence in current_doc.sents]

# Get the entities
detected_entities = [(entity.text, entity.label_, str(spacy.explain(entity.label_))) for entity in current_doc.ents]

# Get the nouns
detected_nouns = [this_noun.text for this_noun in current_doc.noun_chunks]

# Stemming refers to reducing a word to its root form
# Stemming is a text normalization technique that cuts off the end or beginning of a word by
# taking into account a list of common prefixes or suffixes that could be found in that word.
# It is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
stemmed_text = [stemmer.stem(word.text) for word in current_doc if not word in all_stopwords]