'''

HappyTransformer_tutorial.py

# https://happytransformer.com/
Happy Transformer is a package built on top of Hugging Face’s transformer library
that makes it easy to utilize state-of-the-art NLP models.


pip3 install happytransformer

- Google has released the weights for T5 (https://huggingface.co/models)

Fine tune one of these models for grammar correction:
https://www.vennify.ai/fine-tune-grammar-correction/


Gramformer (https://github.com/PrithivirajDamodaran/Gramformer)

'''


# Happy Transformer is built on top of Hugging Face’s Transformers library to make it easier to use.
from happytransformer import HappyTextToText, TTSettings



settings = TTSettings(do_sample=True, top_k=10, \
  temperature=0.5, min_length=1, max_length=100)

# ----------
# Grammar correction
# ----------

# 1- Interface with Gramformer
#
# Use a transformer model called Gramformer to correct text
# T5 models are able to perform multiple tasks with a single model.
happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1")

# For the Gramformer model we’re using, the only prefix we need is “gec:.”
text = "gec: " + "I likes to eatt. applees"



result = happy_tt.generate_text(text, args=settings)
print(result.text)


# 2- alternatives
# https://www.vennify.ai/top-t5-transformer-models/
happy_grammar = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

text = "I like to walks my dog."
input_text = "grammar: " + text
result = happy_grammar.generate_text(input_text, args=beam_args)

print(result.text)




# ----------
# 
# ----------
# >> mrm8488/t5-base-finetuned-common_gen

'''
Take in a set of words and then produce text based on the words provided. 
So, perhaps you have a few keywords for text you wish to produce, 
then you can use this model to generate text relating to those keywords.
'''

happy_common_gen = HappyTextToText("T5", "mrm8488/t5-base-finetuned-common_gen")
beam_args = TTSettings(num_beams=5, min_length=1, max_length=100)

# from here: https://ieeexplore.ieee.org/abstract/document/9363114/keywords#keywords
input_keywords = "sales, machine learning,retail promotions,supply chain".replace(',', ' ')
result = happy_common_gen.generate_text(input_keywords, args=beam_args)
print(result.text)
