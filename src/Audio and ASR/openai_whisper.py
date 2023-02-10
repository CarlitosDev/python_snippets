'''

pip3 install tokenizers --upgrade
pip3 install git+https://github.com/openai/whisper.git 
pip3 install ffmpeg-python --upgrade

We observed that the difference becomes less significant for the small.en and medium.en models

whisper /Users/carlos.aguilar/Documents/ASR/audio_testing/90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav --model small.en

path to download the model files; by default, it uses "~/.cache/whisper"



https://openai.com/blog/whisper/

Good discussion:
https://www.assemblyai.com/blog/how-to-run-openais-whisper-speech-recognition-model/

'''

import whisper
import os
model_name = 'small.en'
model_name = 'small'
base_folder = '/Users/carlos.aguilar/Documents/ASR/audio_testing/'
base_folder = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/ASR/audio_testing'
audio_file = os.path.join(base_folder, '90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav')

model = whisper.load_model(model_name)

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)


#####
model_name = 'small.en'
model = whisper.load_model(model_name)
import numpy as np
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)
options = whisper.DecodingOptions(language="en", without_timestamps=True)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
results = model.decode(mel, options)

## For some reason it works with the runtime.
# source ~/.bash_profile;
# whisper 90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav --model base --language en --output_dir "/Volumes/TheStorageSaver/29.12.2021-EducationFirst/ASR/audio_testing/"
# !whisper /content/90f547b9-176b-4e0d-96f7-ae78cac0b497b_16kHz.wav --model medium.en --language en --no_speech_threshold 0.9 --output_dir /content/whisper

# inside of an EC2 instance
mkdir -p data/whisper
whisper /root/data/2041566945.flac --model small.en --language en --no_speech_threshold 0.9 --output_dir /root/data/whisper
# EC2/Python
model = whisper.load_model(model_name)
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio('/root/data/2041566945.flac')
options = dict(language="en", without_timestamps=False)
results = whisper.transcribe(model, audio, verbose=True, no_speech_threshold=0.9)
results['segments'][3]
# avg_logprob is the result of every segment, which is the average of log softmax of the logits.
# Alternatively, change the update() method by adding
# token_logits = torch.stack([logits[k, next_tokens[k]] for k in range(next_tokens .shape[0])], dim=0)

### Testing with Transformers

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")


import carlos_utils.audio_utils as adu
spk_A, Fs_A = adu.read_audio_file(local_path_to_sample)

inputs = processor(spk_A, return_tensors="pt", sampling_rate=Fs_A)
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
transcription




spk_B, Fs_B = adu.read_audio_file(local_path_to_audiofile)
if Fs_B != 16000:
     spk_B = adu.resample_audio(spk_B, Fs=Fs_B, new_Fs=16000)
     Fs_B = 16000

inputs = processor(spk_B, return_tensors="pt", sampling_rate=Fs_B)
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
transcription



###
ffprobe 2ceb6c98-f2aa-4e8d-a2e4-dc0666e99700b.flac
ffmpeg -i 2ceb6c98-f2aa-4e8d-a2e4-dc0666e99700b.flac -map_channel 0.1.0 2ceb6c98-f2aa-4e8d-a2e4-dc0666e99700b_0.flac -map_channel 0.1.1 ch1.wav -map_channel 0.1.2 ch2.wav
# !whisper 2ceb6c98-f2aa-4e8d-a2e4-dc0666e99700b.flac --model medium.en --language en --no_speech_threshold 0.9 --output_dir /whisper
--device mps --fp16 False

mkdir whisper_GPU
PYTORCH_ENABLE_MPS_FALLBACK=1 whisper 2ceb6c98-f2aa-4e8d-a2e4-dc0666e99700b.flac --model medium.en --language en --no_speech_threshold 0.9 --device mps --fp16 False --output_dir ./whisper_GPU/
