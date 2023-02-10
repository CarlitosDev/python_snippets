'''
  spacy_spancat.py

  Look up a tutorial as I have no clue about using this.

'''

import spacy
nlp = spacy.load("en_core_web_sm")

from spacy.pipeline.spancat import DEFAULT_SPANCAT_MODEL
config = {
    "threshold": 0.5,
    "spans_key": "labeled_spans",
    "max_positive": None,
    "model": DEFAULT_SPANCAT_MODEL,
    "suggester": {"@misc": "spacy.ngram_suggester.v1", "sizes": [1, 2, 3]},
}

spancat = nlp.add_pipe("spancat", config=config)



# spancat_nlp = spacy.load(spancat_model_path)


doc = nlp("This is a sentence.")
spancat = nlp.add_pipe("spancat")
# This usually happens under the hood
processed = spancat(doc)



# Construction via add_pipe with default model
spancat = nlp.add_pipe("spancat")

# Construction via add_pipe with custom model
config = {"model": {"@architectures": "my_spancat"}}
parser = nlp.add_pipe("spancat", config=config)

# Construction from class
from spacy.pipeline import SpanCategorizer
spancat = SpanCategorizer(nlp.vocab, model, suggester)