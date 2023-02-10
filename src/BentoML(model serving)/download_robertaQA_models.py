# Adapted from https://github.com/bentoml/BentoML/blob/main/examples/custom_runner/nltk_pretrained_model/download_nltk_models.py

import os
from transformers import pipeline

nlp_qa = pipeline('question-answering', model="deepset/roberta-base-squad2")

# #!/usr/bin/env python
# import os
# import nltk

# if os.environ.get("BENTO_PATH"):
#     # Bento setup_script is executed as root user during containerize. By default, NLTK
#     # will download data files to "/root/nltk_data". However, the default user when running
#     # a BentoML generated docker image is "bentoml", which does not have the permission
#     # for accessing "/root".
#     # This ensures NLTK data are downloaded to the "bentoml" user's home directory,
#     # which can be accessed by user code.
#     download_dir = os.path.expandvars("/home/bentoml/nltk_data")
# else:
#     download_dir = os.path.expandvars("$HOME/nltk_data")

# nltk.download("vader_lexicon", download_dir)
# nltk.download("punkt", download_dir)