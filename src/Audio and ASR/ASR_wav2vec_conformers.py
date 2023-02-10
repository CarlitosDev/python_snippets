'''

ASR_wav2vec_conformers.py

# https://huggingface.co/docs/transformers/v4.20.0/en/model_doc/wav2vec2-conformer

pip3 install transformers --upgrade
'''

from locale import currency
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerModel, Wav2Vec2ConformerForCTC, Wav2Vec2ConformerConfig
import torch

configuration = Wav2Vec2ConformerConfig()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
# model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
# Wav2Vec2Conformer Model with a language modeling head on top for Connectionist Temporal Classification (CTC).
model_ctc = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")


# Replace this by my own audio
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
# audio_extract = dataset[0]["audio"]["array"]

import wave
import soundfile as sf

path_to_wavfile = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/ASR/audio_testing/90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav'
wf = wave.open(path_to_wavfile, "rb")
sampling_freq = wf.getframerate()

# wf_snd = sf.SoundFile(path_to_wavfile, 'r+')
audio_data, samplerate = sf.read(path_to_wavfile)

block_size = int(sampling_freq/4)

audio = wf.readframes(block_size)
audio_extract = audio_data[0:block_size*4]



import io
import soundfile as sf
import time
# data, samplerate = sf.read(io.BytesIO(audio_extract))
# data, samplerate = sf.read(audio_extract)


# import numpy as np
# audio_array = np.frombuffer(audio_extract, dtype=np.int16)

# import io
# a = io.BytesIO(audio_extract)

# audio file is decoded on the fly
inputs = processor(audio_extract, sampling_rate=sampling_freq, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

# I don't think this is right...
logits = last_hidden_states
predicted_ids = torch.argmax(logits, dim=-1)
transcription = [processor.decode(predicted_id) for predicted_id in predicted_ids]



# this works
transcription_blocks = []
block_size = int(sampling_freq*8)
block_overlap = int(block_size/4)
for block in sf.blocks(path_to_wavfile, blocksize=block_size, overlap=block_overlap):
  s = time.perf_counter()
  inputs = processor(block, sampling_rate=sampling_freq, return_tensors="pt")
  with torch.no_grad():
      logits = model_ctc(**inputs).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  current_transcription = processor.batch_decode(predicted_ids)
  elapsed = time.perf_counter() - s
  print(f"took {elapsed:0.2f} seconds.")
  print(current_transcription)
  transcription_blocks.append(current_transcription)



# block *4 :: took 3.20 seconds.
# block *6 :: took 4.62 seconds.
# block *10 :: took 8.00 seconds.


'''
block_size = int(sampling_freq*4)
block_overlap = int(block_size/4)

took 3.26 seconds.
['WRITTEN DOWN AND YOU REMEMBER THEM THEY CAN BE USED FOR']
took 3.23 seconds.
['BE USED FOR ANY MEETINGS EVEN THE MOST COMPLICA']
took 3.18 seconds.
['MOST COMPLICATED ONES IF YOU GO LIKE A FINANCIAL REPOR']
took 3.16 seconds.
['FINANCIAL REPORIG WITH ALL THE AMERICAN BOSSESI']
took 3.19 seconds.
['BOSES WRITIN EVEN IF IS THEA SMALL BEATING JUST YOU AND']
took 3.17 seconds.
["MEETING JUST YOU AND YOUR BOSS AND YOU CAN JUST SAY O K SO TO DAY'S AGEIN"]
took 3.20 seconds.
["SO TODAY'S AGENDER WE GOT THERE ARE THREE ITEMS AT FIRST"]
took 3.17 seconds.
['AT FIRST WE CAN TALK ABOUT THIS THEN WE CAN DO THAT FINA']
took 3.21 seconds.
['THAT FINALLY WE CAN SO JIST PRA']
took 3.18 seconds.
["JUST PRACTISE THESE ONES AND YOU'LL BE PERFECT AT MEAT"]
took 3.19 seconds.
['TEKDAP MEETINGS HOISEONLY A VAGO YO']
took 3.18 seconds.
['YO WELL I THINK THAT YOU ARE']
took 3.29 seconds.
['TAT YOUARE READY NOW TO GO TO A MEETING']
took 3.18 seconds.
['A MEETING WHAT DO YOU THINK SHALL WE PRACTISE H AL']
took 2.94 seconds.
["HYR ALLA GON A YRE LAT'S DO IT"]
'''


'''
block_size = int(sampling_freq*8)
block_overlap = int(block_size/4)

took 6.63 seconds.
['WRITTEN DOWN AND YOU REMEMBER THEM THEY CAN BE USED FOR ANY MEETINGS EVEN THE MOST COMPLICATED ONES']
took 6.81 seconds.
['MOST COMPLICATED ONES IF YOU GO LIKE A FINANCIAL REPORT TAK THRO WITH ALL THE AMERICAN BOSSES RIGHT EVEN IF']
took 6.51 seconds.
["BOSSES RIGHT EVEN IF IT'S THE SMALL BEATING JUST YOU AND YOUR BOSS AND YOU CAN JUST SAY O K SOON TO DAYS OR GENDAR WE GO THRO THE A"]
took 6.47 seconds.
["SO TODAY'S AGENDER WE GOT THERE ARE THREE ITEMS AT FIRST WE CAN TALK ABOUT THIS THEN WE CAN DO THAT FINALLY WE CAN"]
took 6.44 seconds.
["THAT FINALLY WE CAN SO JUST PRACTISE THESE ONES AND YOU'LL BE PATIENT AT MEETINGS ILES"]
took 6.50 seconds.
['THAT MEETINGS HAS WHOTDO YOU THINK O YOU WELL I THINK THAT YOU ARE READY NO']
took 6.45 seconds.
['THAT YOUR READY NOW TO GO TO A MEETING WHAT DO YOU THINK SHALL WE PRACTISE Y ALANGONA Y']
took 2.93 seconds.
["HYR ALLA GON A YRE LAT'S DO IT"]

'''