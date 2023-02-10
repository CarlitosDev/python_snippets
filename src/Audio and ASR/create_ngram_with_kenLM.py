'''
  create_ngram_with_kenLM.py

'''

import torch
import os
import json
import numpy as np
from jiwer import wer
import pyperclip as pp
import carlos_utils.file_utils as fu

from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1')
train_text = dataset['train']

# from here: https://huggingface.co/blog/wav2vec2-with-ngram

model_folder = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/ASR/language-models'
english_corpus_file = fu.fullfile(model_folder, 'wikitext-103-v1.txt')
with open(english_corpus_file, "w") as file:
  file.write(" ".join(train_text["text"]))


# in the terminal

'''
cd /Users/carlos.aguilar/Documents/ASR/language-models

/Users/carlos.aguilar/Documents/EF_repos/kenLM/kenlm/build/bin/lmplz -o 5 <"wikitext-103-v1.txt"> "wikitext-103-v1-5gram.arpa" --skip_symbols
/Users/carlos.aguilar/Documents/EF_repos/kenLM/kenlm/build/bin/lmplz -o 4 <"wikitext-103-v1.txt"> "wikitext-103-v1-4gram.arpa" --skip_symbols
/Users/carlos.aguilar/Documents/EF_repos/kenLM/kenlm/build/bin/lmplz -o 3 <"wikitext-103-v1.txt"> "wikitext-103-v1-3gram.arpa" --skip_symbols

'''


# code from https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode/blob/main/fix_lm.py
source_file = fu.fullfile(model_folder, 'wikitext-103-v1-4gram.arpa')
destination_file = source_file.replace('gram.arpa', 'gram_fixed.arpa')

original = open(source_file, 'r').readlines()
fixed = open(destination_file, 'w')

for line in original: 
    if 'ngram 1=' in line:
        base_ngram_1_line = line 
        text, value = line.split('=')
        value = str(float(value.replace('\n', ''))+1)
        fixed_ngram_1_line = f"{text}={value}\n"
        fixed.write(fixed_ngram_1_line)
    elif '\t<s>\t' in line: 
        base_token_line = line 
        fixed_token_line = line.replace('\t<s>\t', '\t</s>\t')
        fixed.write(base_token_line)
        fixed.write(fixed_token_line)
    else:
        fixed.write(line)
fixed.close()

# Alternatively:
# wget https://raw.githubusercontent.com/patrickvonplaten/Wav2Vec2_PyCTCDecode/main/fix_lm.py
# chmod +x fix_lm.py
# ./fix_lm.py --path_to_ngram wikitext-103-v1-4gram.arpa --path_to_fixed wikitext-103-v1-4gram_fixed.arpa
# ./fix_lm.py --path_to_ngram wikitext-103-v1-3gram.arpa --path_to_fixed wikitext-103-v1-3gram_fixed.arpa


## Initialise a decoder
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="5gram_correct.arpa",
)



english_corpus_file = fu.fullfile(model_folder, 'wikitext-103-v1.txt')
english_corpus_file = fu.fullfile(model_folder, 'wikitext-103-v1.txt')


from pyctcdecode import build_ctcdecoder
unigrams_file = open("language_model/vocabulary.txt", "r")
unigrams_list = unigrams_file.readlines()
decoder = build_ctcdecoder(
    labels=list(vocab_dict.keys()),
    kenlm_model_path="language_model/5gram.bin",
    unigrams=unigrams_list
)

from transformers import Wav2Vec2ProcessorWithLM
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model=processor_with_lm, device=0)