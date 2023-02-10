
'''
alphacephei_vosk

https://alphacephei.com/vosk/
https://github.com/alphacep/vosk-api


Installation
https://alphacephei.com/vosk/install

# Let's see if it works..
pip3 install vosk
vosk-transcriber --list-languages
Python examples:
https://github.com/alphacep/vosk-api/tree/master/python/example



#From git
git clone https://github.com/alphacep/vosk-api.git


# Websocket Server and GRPC server
docker run -d -p 2700:2700 alphacep/kaldi-en:latest
https://github.com/alphacep/vosk-server


Video:
https://youtu.be/Itic1lFc4Gg


Models:
https://alphacephei.com/vosk/models

'''

# pip3 install vosk


import wave
import json

# Test the Docker image
# import asyncio
# import websockets
# import sys
# import wave
# import json

# async def run_test(uri):
#     async with websockets.connect(uri) as websocket:

#         wf = wave.open(sys.argv[1], "rb")
#         await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
#         buffer_size = int(wf.getframerate() * 0.2) # 0.2 seconds of audio
#         while True:
#             data = wf.readframes(buffer_size)

#             if len(data) == 0:
#                 break

#             await websocket.send(data)
#             print (await websocket.recv())

#         await websocket.send('{"eof" : 1}')
#         print (await websocket.recv())

# asyncio.run(run_test('ws://localhost:2700'))




####

path_to_wavfile = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/ASR/audio_testing/90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav'
path_to_wavfile = '/Users/carlos.aguilar/Documents/ASR/audio_testing/90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav'


from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave

# You can set log level to -1 to disable debug messages
SetLogLevel(0)

model_folder = '/Users/carlos.aguilar/Documents/ASR/VOSK_models'
# 40MB
current_model = 'vosk-model-small-en-us-0.15'
# 1.0G
current_model = 'vosk-model-en-us-daanzu-20200905'
# 1.8G
current_model = 'vosk-model-en-us-0.22'
# Dynamic graph (128MB)
# current_model = 'vosk-model-en-us-0.22-lgraph'

path_to_model = os.path.join(model_folder, current_model)

model = Model(model_path=path_to_model, model_name=current_model)

# model = Model(lang="en-us")


wf = wave.open(path_to_wavfile, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print ("Audio file must be WAV format mono PCM.")
    exit (1)



# You can also init model by name or with a folder path
# model = Model(model_name="vosk-model-en-us-0.21")
# model = Model("models/en")

sampling_freq = wf.getframerate()
rec = KaldiRecognizer(model, sampling_freq)
rec.SetWords(True)
# this will produce N alternative results (in a list)
#rec.SetMaxAlternatives(2)
rec.SetPartialWords(False)
rec.SetNLSML(True)

local_file_partial = path_to_wavfile.replace('.wav', 'transcript_partial.txt')
local_file_results = path_to_wavfile.replace('.wav', 'transcript_result.txt')

transcription_output_partial = ''
transcription_output_final_raw = ''
transcription_output_final = ''
json_transcription_output = []

iteration=-1
while True:
    # data = wf.readframes(int(sampling_freq/2))
    data = wf.readframes(int(sampling_freq))
    iteration+=1
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        _result = rec.Result()
        if len(_result) > 17:
            json_blob = json.loads(_result)
            json_transcription_output.append(json_blob)
            transcription_output_final_raw +=_result
            transcription_output_final+=json_blob.get('text', '')
    else:
        _partial = rec.PartialResult()
        if len(_partial) > 20:
            transcription_output_partial+=_partial


# with open(local_file_partial, 'w') as f:
#     f.write(transcription_output_partial)
# with open(local_file_results, 'w') as f:
#     f.write(transcription_output_final)

local_json_results = path_to_wavfile.replace('.wav', 'transcript_result.json')
import carlos_utils.file_utils as fu
fu.writeJSONFile(json_transcription_output, local_json_results)

print(rec.FinalResult())



## Skinny version
wf = wave.open(path_to_wavfile, "rb")
sampling_freq = wf.getframerate()
rec = KaldiRecognizer(model, sampling_freq)
rec.SetWords(True)
rec.SetPartialWords(False)
rec.SetNLSML(True)

while True:
    data = wf.readframes(int(sampling_freq/4))
    iteration+=1
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        _result = rec.Result()
        if len(_result) > 17:
            json_transcription_output.append(json.loads(_result))
json_transcription_output.append(json.loads(rec.FinalResult()))


# convert into subtitles so they can be added to the video.
import datetime
subs = []
import str
WORDS_PER_LINE = 7
for i, jres in enumerate(json_transcription_output):
    if not 'result' in jres:
        continue
    words = jres['result']
    for j in range(0, len(words), WORDS_PER_LINE):
        line = words[j : j + WORDS_PER_LINE] 
        s = srt.Subtitle(index=len(subs), 
                content=" ".join([l['word'] for l in line]),
                start=datetime.timedelta(seconds=line[0]['start']), 
                end=datetime.timedelta(seconds=line[-1]['end']))
        subs.append(s)
print(srt.compose(subs))




# from transformers import AutoTokenizer, AutoModelForTokenClassification
# tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
# model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")

# idx = 0
# text = json_transcription_output[idx]['text']
# inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
# inputs['input_ids'].shape
# with torch.no_grad():
#     logits = model(**inputs).logits

# # WHY 38 instead of 31??
# predicted_token_class_ids = logits.argmax(-1)
# predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
# predicted_tokens_classes




text = ''
for this_blob in json_transcription_output:
    text += this_blob['text'] + ' '

# pip3 install deepmultilingualpunctuation
from deepmultilingualpunctuation import PunctuationModel
pt_model = PunctuationModel()
result = pt_model.restore_punctuation(text)
print(result)

clean_text = pt_model.preprocess(text)
lab_words = pt_model.predict(clean_text)
# len(lab_words)


# json_transcription_output

clean_text = model.preprocess(text)
labled_words = model.predict(clean_text)








# from transformers import RobertaTokenizer, RobertaForTokenClassification
# import torch

# tokenizer = RobertaTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# model = RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

# inputs = tokenizer(
#     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
# )

# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_token_class_ids = logits.argmax(-1)

# # Note that tokens are classified rather then input words which means that
# # there might be more predicted token classes than words.
# # Multiple token classes might account for the same word
# predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
# predicted_tokens_classes




# pip3 install deepmultilingualpunctuation
from deepmultilingualpunctuation import PunctuationModel
pt_model = PunctuationModel()

## Skinny version
wf = wave.open(path_to_wavfile, "rb")
sampling_freq = wf.getframerate()
rec = KaldiRecognizer(model, sampling_freq)
rec.SetWords(True)
rec.SetPartialWords(False)
rec.SetNLSML(True)

while True:
    data = wf.readframes(int(sampling_freq/4))
    iteration+=1
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        _result = rec.Result()
        if len(_result) > 17:
            json_transcription_output.append(json.loads(_result))
# json_transcription_output.append(json.loads(rec.FinalResult()))

# Format as one piece of text.
transcription = ''
for this_blob in json_transcription_output:
    transcription += this_blob['text'] + ' '


transcription_punctuation = pt_model.restore_punctuation(transcription)

clean_text = pt_model.preprocess(transcription)
lab_words = pt_model.predict(clean_text)

transcription_utterances = []
idx_utterance = -1
this_blob = json_transcription_output[3]
for this_blob in json_transcription_output:
    for this_utterance in this_blob['result']:
        idx_utterance += 1
        if (this_utterance['word'] == lab_words[idx_utterance][0]) and lab_words[idx_utterance][1] != '0':
            this_utterance['word_with_punctuation'] = this_utterance['word'] + lab_words[idx_utterance][1]
        else:
            this_utterance['word_with_punctuation'] = this_utterance['word']
        transcription_utterances.append(this_utterance)




# Try VOSK directly on the videos

import subprocess
video_sampling_freq=48000
print_frequency = 16
iteration =-1
json_transcription_output = []
videofile = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/29.06.2022/90f547b9-176b-4e0d-96f7-ae78cac0b497/evc_API/2033696474.mp4'
process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                            videofile,
                            '-ar', str(video_sampling_freq) , '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE)

block_size = int(video_sampling_freq/4)
while True:
    data = process.stdout.read(block_size)
    iteration+=1
    if iteration % print_frequency==0:
        print(f'Processing second {iteration/4} ')
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        _result = rec.Result()
        if len(_result) > 17:
            json_transcription_output.append(json.loads(_result))
# json_transcription_output.append(json.loads(rec.FinalResult()))

json_file = videofile.replace('.mp4', '.json')
import carlos_utils.file_utils as fu
fu.writeJSONFile(json_transcription_output, json_file)