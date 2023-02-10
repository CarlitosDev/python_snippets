'''
	Spacy_detect_idioms.py
'''


# idioms 

Spacy implementation
https://github.com/eubinecto/idiomatch


>> Installation
python3 -m spacy download en_core_web_sm
Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl (13.6 MB)


# idioms - it does not work very well.
# it piggybacks on https://spacy.io/api/matcher
from spacy import load as spacy_load
nlp_web_sm = spacy_load("en_core_web_sm")  # idiom matcher needs an nlp pipeline; Currently supports en_core_web_sm only.

from idiomatch import Idiomatcher
idiomatcher = Idiomatcher.from_pretrained(nlp_web_sm)  # this will take approx 50 seconds.

doc = nlp_web_sm(transcription)  # process the sentence with an nlp pipeline

idioms_found = idiomatcher.identify(doc)
print(idiomatcher.identify(doc))  # identify the idiom in the sentence



Paper
https://www.mdpi.com/1019008

Stanford
https://nlp.stanford.edu/~muzny/docs/mz-emnlp2013.pdf