'''

nlp_for_English.py


  08.03.2022 - Some trivial questions that I still don't know the answer to.
  1 - NLP to differenciate a well structured sentence from a bad one???



'''
import carlos_utils.file_utils as fu
import carlos_utils.nlp_utils as nlpu



sentence_A = 'How to differenciate a well structured sentence from a bad one?'
sentence_B = 'How to differentiate a well structured sentence from a bad one?'






#pip3 install happytransformer
from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5",  "prithivida/grammar_error_correcter_v1")

settings = TTSettings(do_sample=True, top_k=10, \
  temperature=0.5, min_length=1, max_length=100)

text = "gec: " + sentence_A
result = happy_tt.generate_text(text, args=settings)
print(result.text)
# this is actually pretty good:
# How to distinguish a well structured sentence from a bad one?


# https://www.vennify.ai/top-t5-transformer-models/
happy_grammar = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

beam_args = TTSettings(num_beams=5, min_length=1, max_length=100)

input_text = "grammar: " + sentence_A
result = happy_grammar.generate_text(input_text, args=beam_args)

print(result.text)





path_to_picke = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons/09.02.2022/8f210358-5c68-463a-83a8-6e1d63688bea/lesson_analysis/analysis_8f210358-5c68-463a-83a8-6e1d63688bea.pickle'
transcription_analysis = fu.readPickleFile(path_to_picke)
df_transcribe_ext = transcription_analysis.get('df_transcribe_ext')
lesson_summary = transcription_analysis['lesson_summary']
df_speaker_conversational_chunk = transcription_analysis['df_speaker_conversational_chunk']
df_speaker_conversational_chunk.tagged_speaker = df_speaker_conversational_chunk.tagged_speaker.astype(str)

df_speaker_conversational_chunk.iloc[32]


df_speaker = df_speaker_conversational_chunk.groupby(df_speaker_conversational_chunk['tagged_speaker'])
speakers = [*df_speaker.groups.keys()]



idx = 1
speaker_sentences = ' '.join(df_speaker.get_group(speakers[idx]).sentences.tolist()).replace('[', '').replace(']', '')
from carlos_utils.textInspector_utils import get_text_inspector_analysis
cefr_analysis = get_text_inspector_analysis(speaker_sentences, textMode='Listening')

fu.printJSON(cefr_analysis)
folder, filename, _ = fu.fileparts(path_to_picke)
cefr_file = fu.fullfile(folder, f'CEFR_analysis_{speakers[idx]}.json')
fu.writeJSONFile(cefr_analysis, cefr_file)



# another lesson??
path_to_picke = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons/15.02.2022/d6f56230-822d-4729-8f3b-163874f95e6e/lesson_analysis/analysis_d6f56230-822d-4729-8f3b-163874f95e6e.pickle'
transcription_analysis = fu.readPickleFile(path_to_picke)
df_transcribe_ext = transcription_analysis.get('df_transcribe_ext')
lesson_summary = transcription_analysis['lesson_summary']
df_speaker_conversational_chunk = transcription_analysis['df_speaker_conversational_chunk']
df_speaker_conversational_chunk.tagged_speaker = df_speaker_conversational_chunk.tagged_speaker.astype(str)

folder, filename, _ = fu.fileparts(path_to_picke)
for this_speaker, df_speaker in df_speaker_conversational_chunk.groupby(df_speaker_conversational_chunk['tagged_speaker']):
  speaker_sentences = ' '.join(df_speaker.sentences.tolist()).replace('[', '').replace(']', '')
  cefr_analysis = get_text_inspector_analysis(speaker_sentences, textMode='Listening')
  cefr_file = fu.fullfile(folder, f'CEFR_analysis_{this_speaker}.json')
  fu.writeJSONFile(cefr_analysis, cefr_file)





'''
  
  pip3 install gingerit
  pip3 uninstall gingerit

It fails...

from gingerit.gingerit import GingerIt

parser = GingerIt()
parser.parse(sentence_A)

'''




path_to_transcription = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons/09.02.2022/8f210358-5c68-463a-83a8-6e1d63688bea/8f210358-5c68-463a-83a8-6e1d63688bea.json'
jsonBlob = fu.readJSONFile(path_to_transcription)

jsonBlob['results'].keys()


['language_code', 'transcripts', 'speaker_labels', 'language_identification', 'items']

speaker_labels = jsonBlob['results']['speaker_labels']
num_speakers = speaker_labels['speakers']
speaker_segments = speaker_labels['segments']
total_segments = len(speaker_segments)

idx_segment = 120
fu.printJSON(speaker_segments[idx_segment])



import pandas as pd
# items
df_speakers = pd.json_normalize(data=speaker_segments, record_path='items')
df_speakers.iloc[3]
df_speakers.iloc[0]
df_speakers.iloc[2]




transcription_items = jsonBlob['results']['items']


idx = 0
for a in transcription_items:
  a
  idx +=1
  if idx==5: break


transcription_items[0]
item_key = f'''{transcription_items[0]['start_time']}_{transcription_items[0]['end_time']}'''
type = transcription_items[0]['type']

content = transcription_items[0]['alternatives'][0]['content']
confidence = transcription_items[0]['alternatives'][0]['confidence']




transcription_components = {}
previous_key = ''
for _item in jsonBlob['results']['items']:
  item_type = _item['type']
  if item_type == 'pronunciation':
    item_key = f'''{_item['start_time']}_{_item['end_time']}'''
    content = _item['alternatives'][0]['content']
    confidence = float(_item['alternatives'][0]['confidence'])
    item_features = {
      'content': content, 
      'content_and_punctuation': content, 
      'confidence': confidence
      }
    transcription_components.update({item_key:item_features})
  else:
    transcription_components[item_key]['content_and_punctuation'] += _item['alternatives'][0]['content']  
  previous_key = item_key



transcription_parsed = []
void_features = {'content': '', 'content_and_punctuation': '', 'confidence': .0}
previous_end_time = float(speaker_segments[0]['items'][0]['start_time'])
for segment_id, this_segment in enumerate(speaker_segments):
  for this_item in this_segment['items']:

    start_time = float(this_item['start_time'])
    end_time = float(this_item['end_time'])

    segment_features = {'segment_id': segment_id, 
    'start_time': start_time,
    'end_time': end_time,
    'duration': end_time-start_time,
    'speaker_label': this_item['speaker_label'],
    'gap_from_previous_word': start_time-previous_end_time}
    
    previous_end_time = start_time

    item_key = f'''{this_item['start_time']}_{this_item['end_time']}'''
    segment_features.update(transcription_components.get(item_key, void_features))

    transcription_parsed.append(segment_features)

df_transcription = pd.DataFrame(transcription_parsed)
df_transcription.iloc[0]
df_transcription.iloc[1]
df_transcription.iloc[2]