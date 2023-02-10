'''
	from_mp3_to_wav.py
'''


mp3_filepath = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videos/24.01.2022/72a48689-e72b-4439-9888-3e792f35aea9/72a48689-e72b-4439-9888-3e792f35aea9b.mp3'
wav_filepath = mp3_filepath.replace('mp3','wav')

from pydub import AudioSegment
audio_object = AudioSegment.from_mp3(mp3_filepath)
# this will do the whole thing...44.1kHz, 16 bits and 2 channels...
# audio_object.export(wav_filepath, format="wav")


audio_object_downsampled = audio_object.set_frame_rate(16000)
# set to 8 bits
audio_object_downsampled = audio_object_downsampled.set_sample_width(1)
audio_object_downsampled.export(wav_filepath, format="wav")