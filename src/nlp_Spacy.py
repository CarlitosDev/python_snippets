pip3 install -U spacy
python3 -m spacy download en
python3 -m spacy download es

Good tutorial: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/



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