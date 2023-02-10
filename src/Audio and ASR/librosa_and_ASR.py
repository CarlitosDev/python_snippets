'''
.py

source ~/.bash_profile && python3 librosa_and_ASR.py
'''



import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf


new_fs = 16000
file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/26c86a91-ef03-433c-bd12-98880c4e6399b.mp3'
spk_A, Fs_A = librosa.load(file_A, sr=None)
y_spk_A_16k = librosa.resample(spk_A, orig_sr=Fs_A, res_type='kaiser_fast', target_sr=new_fs)

file_B = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/1039006966.mp4'
spk_B, Fs_B = librosa.load(file_B, sr=None)
y_spk_B_16k = librosa.resample(spk_B, orig_sr=Fs_B, res_type='kaiser_fast', target_sr=new_fs)



# manually
# The number of samples per window
window_seconds = 1
frame_length = int(new_fs*window_seconds)
# The number of samples between analysis frames
hop_length = int(new_fs*0.5)

split_signal_A = librosa.effects.split(y=y_spk_A_16k,  frame_length=frame_length, hop_length=hop_length, top_db=10)

num_intervals = 4
y_spk_A_16k_chunk = y_spk_A_16k[split_signal_A[0][0]:split_signal_A[num_intervals][1]]

y_spk_A_16k_chunk = None
min_speech_seconds = 3
for this_interval in split_signal_A:
  duration_interval = (this_interval[1]-this_interval[0])/new_fs
  if duration_interval>=min_speech_seconds:
      y_spk_A_16k_chunk = y_spk_A_16k[this_interval[0]:this_interval[1]]
      break
file_C = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_A_16k_chunk.wav'
sf.write(file_C, y_spk_A_16k_chunk, new_fs)



split_signal_B = librosa.effects.split(y=y_spk_B_16k,  frame_length=frame_length, hop_length=hop_length, top_db=10)


y_spk_B_16k_chunk = None
min_speech_seconds = 3
for this_interval in split_signal_B:
  duration_interval = (this_interval[1]-this_interval[0])/new_fs
  if duration_interval>=min_speech_seconds:
      y_spk_B_16k_chunk = y_spk_B_16k[this_interval[0]:this_interval[1]]
      break


file_D = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_B_16k_chunk.wav'
sf.write(file_D, y_spk_B_16k_chunk, new_fs)

plt.plot(y_spk_B_16k_chunk)
plt.show()


# import time
# sd.play(y_spk_A_16k_chunk, new_fs)
# time.sleep(12)
# sd.stop()


# Wav2Vec2ForCTC, which allows to restore pretrained models and train them with the Connectionist Temporal Classification (CTC)
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

# #Loading the model and the tokenizer
# model_name = "facebook/wav2vec2-base-960h"
# tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)

# #Tokenize
# input_values = tokenizer(y_spk_A_16k_chunk, return_tensors="pt").input_values
# #Take logits
# logits = model(input_values).logits
# #Take argmax
# predicted_ids = torch.argmax(logits, dim=-1)
# #Get the words from predicted word ids
# transcription = tokenizer.decode(predicted_ids[0])
# print(transcription)
'''
SO AWAY FROM  MARGERI DSPI OCCASO WONHELANED AN I CLAF STOPPING YONE SECOND PLES O WHAT PROWEATHER ON THIS TANE I LEAVE HIM MADRIK T ALL THE TRAIN IN TO DAY SAY
'''

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-base-ls960")
# model = AutoModel.from_pretrained("facebook/hubert-base-ls960")

# from transformers import AutoTokenizer, AutoModelForCTC
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")



# from transformers import AutoTokenizer, AutoModelForCTC
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")




# A
# from transformers import Wav2Vec2Processor, HubertForCTC
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
# model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# input_values = processor(y_spk_A_16k_chunk, return_tensors="pt", sampling_rate=new_fs).input_values
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.decode(predicted_ids[0])

# print(transcription)


# using facebook/hubert-large-ls960-ft (A)
'''
SO WAYFROM YOU AN WA MAGERY GOSPI OC SO WAER CALANI I MEAN CLASS STOP IT ONE SECOND PLEASE O WHAT' THE WEATHER ON THIS PAIN I LIVE IN MADRID THE AL  IS RAINING TO DAY SAR
'''



# pip3 install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip 
# from transformers import Wav2Vec2ProcessorWithLM
# processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
# inputs = processor(audio_sample["audio"]["array"], sampling_rate=16_000, return_tensors="pt")
# transcription = processor.batch_decode(logits.numpy()).text
# transcription[0].lower()

# LDFLAGS="-L/opt/local/lib" CFLAGS="-I/opt/local/include" pip3 install https://github.com/kpu/kenlm/archive/master.zip
# from transformers import AutoProcessor, AutoModelForCTC
# processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
# model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

import os
import librosa
import soundfile as sf
import torch
file_C = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_A_16k_chunk.wav'
y_spk_A_16k_chunk, Fs = sf.read(file_C)

# from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
# model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
# processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
# input_features = processor(y_spk_A_16k_chunk, sampling_rate=16_000, return_tensors="pt").input_features
# crashes here...
# generated_ids = model.generate(input_ids=input_features)

# transcription = processor.batch_decode(generated_ids)


from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

inputs = processor(y_spk_A_16k_chunk, sampling_rate=Fs, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("Prediction:", predicted_sentence)
'''
so ware frommargerite spile okay saw what kalany unin class stop it one second please no what's the weather along this fam i lve in madridwith raining day sa
'''

#https://huggingface.co/spaces/Cahlil/Speech-Recognition-with-Speaker-Segmentation/blob/main/app.py

from pyannote.audio import Pipeline
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-large-960h-lv60-self",
    feature_extractor="facebook/wav2vec2-large-960h-lv60-self",
    
)
speaker_segmentation = Pipeline.from_pretrained("pyannote/speaker-segmentation")

def segmentation(audio):
    speaker_output = speaker_segmentation(audio)
    text_output = asr(audio,return_timestamps="word")
    
    full_text = text_output['text'].lower()
    chunks = text_output['chunks']

    diarized_output = ""
    i = 0
    for turn, _, speaker in speaker_output.itertracks(yield_label=True):
        diarized = ""
        while i < len(chunks) and chunks[i]['timestamp'][1] <= turn.end:
            diarized += chunks[i]['text'].lower() + ' '
            i += 1
        
        if diarized != "":
            diarized_output += "{}: ''{}'' from {:.3f}-{:.3f}\n".format(speaker,diarized,turn.start,turn.end)
        
    return diarized_output, full_text







# this looks promising
from transformers import pipeline
import soundfile as sf

file_C = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_A_16k_chunk.wav'

file_D = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_B_16k_chunk.wav'
y_spk_D_16k_chunk, Fs = sf.read(file_D)

asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/hubert-large-ls960-ft",
    feature_extractor="facebook/hubert-large-ls960-ft",
)
text_output = asr(y_spk_D_16k_chunk,return_timestamps="word")
text_output['chunks']
'''
 "SO WAYFROM YOU AN WA MAGERY GOSPI OC SO WAER CALANI I MEAN CLASS STOP IT ONE SECOND PLEASE O WHAT' THE WEATHER ON THIS PAIN I LIVE IN MADRID THE AL  IS RAINING TO DAY SAR"
'''







# this produces the same output as using the torch code above.
from transformers import pipeline
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    feature_extractor=MODEL_ID,
)
# this doesn't return the timestamps??
text_output = asr(y_spk_B_16k_chunk, return_timestamps="word")
'''
"so ware frommargerite spile okay saw what kalany unin class stop it one second please no what's the weather along this fam i lve in madridwith raining day sa",
'''


# this works
# pip3 install huggingsound pyctcdecode --upgrade
# this library gives back the probabilities and the timestamps in miliseconds
file_D = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_B_16k_chunk.wav'
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder, GreedyDecoder
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = [file_D]
transcriptions = model.transcribe(audio_paths)
import carlos_utils.file_utils as fu
fu.printJSON(transcriptions)

# let's try with a Facebook model
model_fcb = SpeechRecognitionModel("facebook/hubert-large-ls960-ft")
audio_paths = [file_C]
transcription_hubert = model_fcb.transcribe(audio_paths)
import carlos_utils.file_utils as fu
fu.printJSON(transcription_hubert)
'''

  "transcription": "so wayfrom you an wa magery gospi oc so waer calani i mean class stop it one second please o what' the weather on this pain i live in madrid the al  is raining to day sar",
'''



# working well
# this looks promising
from transformers import pipeline
import soundfile as sf
import torch
file_C = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/y_spk_A_16k_chunk.wav'
y_spk_A_16k_chunk, Fs = sf.read(file_C)
model_fcb = SpeechRecognitionModel("facebook/hubert-large-ls960-ft")
# do it manually from audio instead of from a file
inputs = model_fcb.processor(y_spk_A_16k_chunk, sampling_rate=Fs, return_tensors="pt", padding=True, do_normalize=True)
with torch.no_grad():
    if hasattr(inputs, "attention_mask"):
        logits = model_fcb.model(inputs.input_values.to(model_fcb.device), attention_mask=inputs.attention_mask.to(model_fcb.device)).logits
    else:
        logits = model_fcb.model(inputs.input_values.to(model_fcb.device)).logits


decoder = GreedyDecoder(model_fcb.token_set)
result_greedy = decoder(logits)
result_greedy[0]['transcription']
# dict_keys(['transcription', 'start_timestamps', 'end_timestamps', 'probabilities'])
#'start_timestamps': [80, 240, 300, 320,

# try with other decoders...
# crashes
lm_path = '/Users/carlos.aguilar/Documents/language_models/mixed-lower.binary'
unigrams_path = '/Users/carlos.aguilar/Documents/language_models/words.mixed_lm.txt'
kensho_decoder = KenshoLMDecoder(model_fcb.token_set, lm_path=lm_path, unigrams_path=unigrams_path)
result_kensho = kensho_decoder(logits)





# The LM format used by the LM decoders is the KenLM format (arpa or binary file).
# You can download some LM files examples from here: 
# https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/main/language_model
lm_path = "/Users/carlos.aguilar/Documents/language_models/lm.binary"
unigrams_path = "/Users/carlos.aguilar/Documents/language_models/unigrams.txt"

# We implemented three different decoders for LM boosted decoding: 
#   KenshoLMDecoder, ParlanceLMDecoder, and FlashlightLMDecoder

# On this example, we'll use the KenshoLMDecoder
# https://github.com/kensho-technologies/pyctcdecode
kensho_decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path)
result_kensho = kensho_decoder(logits)

transcriptions = model.transcribe(audio_paths, decoder=decoder)

print(transcriptions)



lm_path = '/Users/carlos.aguilar/Documents/language_models/mixed-lower.binary'
unigrams_path = '/Users/carlos.aguilar/Documents/language_models/words.mixed_lm.txt'




# this crashes Python...
# pip3 install nemo_toolkit
# import nemo.collections.asr as nemo_asr

# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
#   model_name='stt_en_conformer_ctc_small'
# )
# logits = asr_model.transcribe(["my_file.wav"], logprobs=True)[0]

# decoder = build_ctcdecoder(asr_model.decoder.vocabulary)
# decoder.decode(logits)






# from https://huggingface.co/facebook/hubert-large-ls960-ft
# from transformers import AutoTokenizer, AutoModelForCTC
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")