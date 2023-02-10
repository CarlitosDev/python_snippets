collocations_in_Spacy.py


# https://pypi.org/project/collocater/
# 
# pip3 install collocater --no-deps

from collocater import collocater
import spacy
from pprint import pprint

collie = collocater.Collocater.loader()
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(collie)

text = "If this isn't a bunch of beautiful flowers I don't know what is!"
doc = nlp(text)
print(doc._.collocs) # returns [bunch of, bunch of beautiful flowers, beautiful flowers]

#Tokens with associated collocations in text:
colls = [(col.text, col.start_char, col.end_char, col.label_) for col in doc._.collocs]
pprint(colls) # returns [
#                          ('bunch of', 16, 24, 'bunch_noun__prep'),
#                          ('bunch of beautiful flowers', 16, 42, 'flower_noun__quant'),
#                          ('beautiful flowers', 25, 42, 'flower_noun__adj')
#                          ]


