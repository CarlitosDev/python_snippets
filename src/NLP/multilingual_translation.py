'''
	multilingual_translation.py
 
 
 pip3 install sentencepiece
(https://github.com/google/sentencepiece)
'''

# ???
# from transformers import AutoTokenizer, AutoModelWithLMHead
# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = AutoModelWithLMHead.from_pretrained("t5-base")


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

input = "My name is Azeem and I live in India"

# You can also use "translate English to German"
input_ids = tokenizer("translate English to German: "+input, return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
# You can also use "translate English to French" and "translate English to Romanian"
input_ids = tokenizer("translate English to French: " + input, return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
source_lang = 'English'
target_lang = 'Spanish'
# TODO: look for tutorials on how to use this???
translation_task = f"translate {source_lang} to {target_lang}"
model_inputs = tokenizer(translation_task, return_tensors="pt")
outputs = model.generate(model_inputs.input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-cc25")





# from here:
# https://huggingface.co/transformers/model_doc/marian.html

from transformers import MarianMTModel, MarianTokenizer
src_text = [
    '>>fra<< this is a sentence in english that we want to translate to french',
    '>>por<< This should go to portuguese',
    '>>esp<< And this to Spanish'
]

# https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/eng-roa/README.md
model_name = 'Helsinki-NLP/opus-mt-en-roa'
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)


# codes here:
# https://www.loc.gov/standards/iso639-2/php/code_list.php
'''
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', 
'>>pap<<', '>>ast<<', '>>cat<<', 
'>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', 
'>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', 
'>>arg<<', '>>min<<']
'''

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
[tokenizer.decode(t, skip_special_tokens=True) for t in translated]



# Translate a video.
import carlos_utils.lumiere_analysis_utils as lmu
import carlos_utils.file_utils as fu

this_json_file = '/Users/carlos.aguilar/Documents/EF_AWS_Lumiere/5e7875e79c808d187c32ed379289de4728c25473.json'
json_blob = fu.readJSONFile(this_json_file)

json_blob['results']['speaker_labels'].keys()
json_blob['results'].keys()

transcription_items = json_blob['results']['items']
speaker_segments = json_blob['results']['speaker_labels']['segments']

json_blob['results']['language_identification']
json_blob['results']['language_code']

df_transcribe_ext, conversational_chunks, conversational_speaker_chunks, words_stats = \
  lmu.process_transcribe_and_split_conversation(transcription_items, speaker_segments, \
  timeout_seconds = 1.0, min_confidence = 0.75)
  


# translate sentence by sentence
this_speaker_sentence = conversational_speaker_chunks[2]
fu.printJSON(this_speaker_sentence)
current_sentence = this_speaker_sentence['sentences']
languages_to_translate_to = ['fra', 'por', 'spa', 'ita', 'ron']
src_text = [f'>>{this_lang}<< {current_sentence}' for this_lang in languages_to_translate_to]

# src_text = [
#     f'>>fra<< {current_sentence}',
#     f'>>por<< {current_sentence}',
#     f'>>esp<< {current_sentence}',
#     f'>>ita<< {current_sentence}',
# ]
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
translated_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

translated_sentences_blob = dict(zip(languages_to_translate_to, translated_sentences))

fu.printJSON(translated_sentences_blob)
conversational_chunks[0]


# try with the paragraph
this_paragraph = conversational_chunks[0]
current_sentence = this_paragraph['sentences']
languages_to_translate_to = ['fra', 'por', 'spa', 'ita', 'ron']
src_text = [f'>>{this_lang}<< {current_sentence}' for this_lang in languages_to_translate_to]
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
translated_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

translated_sentences_blob = dict(zip(languages_to_translate_to, translated_sentences))

fu.printJSON(translated_sentences_blob)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")





import carlos_utils.nlp_translation_models as nlptrans
import carlos_utils.lumiere_analysis_utils as lmu
import carlos_utils.file_utils as fu


model = nlptrans.load_marian_multilingual_model()
tokenizer = nlptrans.load_marian_multilingual_tokenizer()


this_json_file = '/Users/carlos.aguilar/Documents/EF_AWS_Lumiere/5e7875e79c808d187c32ed379289de4728c25473.json'
json_blob = fu.readJSONFile(this_json_file)

transcription_items = json_blob['results']['items']
speaker_segments = json_blob['results']['speaker_labels']['segments']

df_transcribe_ext, conversational_chunks, conversational_speaker_chunks, words_stats = \
  lmu.process_transcribe_and_split_conversation(transcription_items, speaker_segments, \
  timeout_seconds = 1.0, min_confidence = 0.75)
  

# translate sentence by sentence
this_speaker_sentence = conversational_speaker_chunks3]
fu.printJSON(this_speaker_sentence)
current_sentence = this_speaker_sentence['sentences']

translated_text = nlptrans.translate_text_marian_MT(current_sentence, model, tokenizer)
fu.printJSON(translated_text)