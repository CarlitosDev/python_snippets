
'''


https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/


Have a look at this one
https://huggingface.co/spaces/reach-vb/asr-pyctcdecode/blob/main/app.py
https://huggingface.co/spaces/DrishtiSharma/ASR_using_Wav2Vec2/blob/main/app.py

'''





mp3_filepath = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons/02.02.2022/eb17ece5-f755-47ad-ae2b-2772369e78b5/eb17ece5-f755-47ad-ae2b-2772369e78b5.mp3'
from pydub import AudioSegment
audio_object = AudioSegment.from_mp3(mp3_filepath)
audio_object_downsampled = audio_object.set_frame_rate(16000)


# pydub does things in milliseconds

import carlos_utils.file_utils as fu
folder, filename, ext = fu.fileparts(mp3_filepath)


min_second = '02:10'
num_seconds = 30
[_min, _sec] = min_second.split(':')
start_time = (int(_min)*60 + int(_sec))*1000
end_time = start_time + num_seconds*1000

extract_conversation = audio_object_downsampled[start_time:end_time]
flacFile = fu.fullfile(folder, f'''{filename}_extract_{min_second.replace(':','.')}_{num_seconds}_seconds.flac''')
extract_conversation.export(flacFile, format = "flac")




from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")


# with open("sample.flac", "rb") as f:
#   data = f.read()

results = pipe(flacFile)
results['text']



# also...https://huggingface.co/facebook/wav2vec2-base-960h


# from transformers import AutoProcessor, AutoModelForCTC
# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")