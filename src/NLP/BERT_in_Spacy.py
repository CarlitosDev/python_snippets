

BERT_in_Spacy.py

examples/Spacy_Transformers_Demo.ipynb


# $ pip install spacy-transformers
# $ python -m spacy download en_trf_bertbaseuncased_lg
# python3 -m spacy download en_trf_bertbaseuncased_lg

import spacy
nlp = spacy.load("en_trf_bertbaseuncased_lg")
apple1 = nlp("Apple shares rose on the news.")
apple2 = nlp("Apple sold fewer iPhones this quarter.")
apple3 = nlp("Apple pie is delicious.")

# sentence similarity
print(apple1.similarity(apple2)) #0.69861203
print(apple1.similarity(apple3)) #0.5404963

# sentence embeddings
apple1.vector  # or apple1.tensor.sum(axis=0)


