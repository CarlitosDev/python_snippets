'''

  delay_time_estimation.py

'''


from pyrsistent import v
from scipy import signal, fftpack
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import carlos_utils.file_utils as fu
import soundfile as sf

# (1) Calculate the cross-correlation between a signal and its delayed version
# read a signal
file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/transcriptions/1039006966_transcription_chunk_0.wav'
spk_A, Fs_A = librosa.load(file_A, sr=None)
num_samples = spk_A.shape[0]

t_axis_seconds = np.arange(num_samples)/Fs_A

delay_in_seconds = 4.32
delay_in_samples = int(delay_in_seconds*Fs_A)

spk_A_delay = np.zeros(len(spk_A)+delay_in_samples)
spk_A_delay[delay_in_samples:] = spk_A

plt.plot(t_axis_seconds, spk_A, alpha=0.5)
plt.plot(np.arange(num_samples+delay_in_samples)/Fs_A, spk_A_delay, color='green', alpha=0.5)
plt.show()


from scipy import signal
correlation = signal.correlate(spk_A, spk_A_delay, mode="full")
lags = signal.correlation_lags(spk_A.size, spk_A_delay.size, mode="full")
lag = lags[np.argmax(correlation)]
lag_seconds = lag/Fs_A
print(f'Shift signal spk_A_delay {lag_seconds} seconds')



# (2) Calculate the cross-correlation between two audio signals
new_fs = 16000
file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/26c86a91-ef03-433c-bd12-98880c4e6399b.mp4'
baseFolder, audioFile_A, audioExt = fu.fileparts(file_A)
spk_A, Fs_A = librosa.load(file_A, sr=None)
y_spk_A_16k = librosa.resample(spk_A, orig_sr=Fs_A, res_type='kaiser_fast', target_sr=new_fs)

file_B = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/1039006966.mp4'
baseFolder, audioFile_B, audioExt = fu.fileparts(file_B)
spk_B, Fs_B = librosa.load(file_B, sr=None)
y_spk_B_16k = librosa.resample(spk_B, orig_sr=Fs_B, res_type='kaiser_fast', target_sr=new_fs)

num_samples = np.min(np.array([y_spk_A_16k.shape[0], y_spk_B_16k.shape[0]])*0.10).astype(int)
num_seconds = num_samples/new_fs

y_A = y_spk_A_16k[0:num_samples]
path_to_audio_extract = fu.fullfile(baseFolder, audioFile_A + '_chunk.wav')
sf.write(path_to_audio_extract, y_A, new_fs)

y_B = y_spk_B_16k[0:num_samples]
path_to_audio_extract = fu.fullfile(baseFolder, audioFile_B + '_chunk.wav')
sf.write(path_to_audio_extract, y_B, new_fs)

from scipy import signal
correlation = signal.correlate(y_A, y_B, mode="full")
lags = signal.correlation_lags(y_A.size, y_B.size, mode="full")
lag_samples = lags[np.argmax(correlation)]
lag_seconds = lag_samples/new_fs
print(f'Shift signal spk_A_delay {lag_seconds} seconds')


t_axis_seconds = np.arange(num_samples)/new_fs
spk_A_delay = np.zeros(num_samples+lag_samples)
spk_A_delay[lag_samples:] = y_B

plt.plot(t_axis_seconds, y_A, alpha=0.5)
plt.plot(np.arange(num_samples+lag_samples)/new_fs, spk_A_delay, color='green', alpha=0.5)
plt.show()




# (3) Calculate the cross-correlation between two audio signals
# find the beginning of the audio file

window_seconds = 1
# The number of samples per analysis frame
frame_length = int(new_fs*window_seconds)
# The number of samples between analysis frames
hop_length = int(new_fs*0.25)
audio_signal_chunks = librosa.effects.split(y=y_spk_A_16k,  frame_length=frame_length, hop_length=hop_length, top_db=10)

num_seconds = np.ceil(len(y_spk_A_16k)/new_fs).astype(int)
steps_per_second = 2
t_half_seconds = np.arange(num_seconds, step=1/steps_per_second)

audio_signal_chunks_seconds = audio_signal_chunks/new_fs
duration_intervals = (audio_signal_chunks[:,1]-audio_signal_chunks[:,0])/new_fs

audio_signal_chunks_seconds_idx = (audio_signal_chunks_seconds*steps_per_second).astype(int)

speech_intervals = np.zeros(shape=len(t_half_seconds),dtype=bool)
for this_interval in audio_signal_chunks_seconds_idx:
  speech_intervals[this_interval[0]:this_interval[1]]=True

t_axis_seconds = np.arange(y_spk_A_16k.shape[0])/new_fs

fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))
ax1 = axes[0]
ax1.plot(t_axis_seconds, y_spk_A_16k, color='blue', alpha=0.6)
ax2 = axes[1]
ax2.plot(t_half_seconds, speech_intervals, color='green', alpha=0.5)
fig.tight_layout()

plt.show()
