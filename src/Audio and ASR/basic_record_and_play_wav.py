'''
basic_record_and_play_wav.py

'''

from scipy.io.wavfile import write as wav_write
def save_audio_piece(audio_matrix: 'numpy.ndarray', filename, sampling_rate):
  wav_write(filename, sampling_rate, audio_matrix)


# pip3 install sounddevice
import sounddevice as sd
sd.play(waveform_quiet, fs)



# record
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 5  # Duration of recording
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  


this_filename = os.path.join(examples_folder, f'carlos_tester_1' + '.wav')
save_audio_piece(myrecording,this_filename,fs)




'''
  Convert formats
  pip3 install pydub
'''
from pydub import AudioSegment
sound = AudioSegment.from_mp3("/path/to/file.mp3")
sound.export("/output/path/file.wav", format="wav")





## Plot a waveform from a video
from pydub import AudioSegment
sound = AudioSegment.from_mp3("/path/to/file.mp3")



import numpy as np
from scipy.io.wavfile import write
samplerate = 44100; fs = 100
t = np.linspace(0., 1., samplerate)
amplitude = np.iinfo(np.int16).max
data_left = amplitude * np.sin(2. * np.pi * fs * t)
data_right = amplitude*0.6 * np.sin(2. * np.pi * fs * t)
audio_data = np.array([data_left.astype(np.int16), data_right.astype(np.int16)]).T
# audio_data = np.vstack((data.astype(np.int16),data.astype(np.int16))).T
audio_data.shape
output_file = '/Users/carlos.aguilar/Documents/tempRubbish/example.wav'
write(output_file, samplerate, audio_data)