'''
librosa_tests.py
'''



import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import librosa.display
import sounddevice as sd

path_to_audio = '/Volumes/GoogleDrive-101555491803780988335/My Drive/Afinación+/AFINACION.wav'


Fs = 44100
x, Fs = librosa.load(path_to_audio, sr=Fs)
print('The signal x has %d samples and a duration of %.2f seconds.' % (len(x), len(x) / Fs))


plt.figure(figsize=(6, 1.8))
plt.plot(x, color='gray')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Define time axis in seconds
t = np.arange(x.shape[0]) / Fs
plt.figure(figsize=(6, 1.8))
plt.plot(t, x, color='gray')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.xlim([t[0], t[-1]])
plt.ylim([-0.40, 0.40])
plt.tick_params(direction='in')
plt.tight_layout()
plt.show()


# amplitude envelope of a waveform
plt.figure(figsize=(6, 1.8))
librosa.display.waveplot(x, sr=Fs, color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()




import librosa.util as lib_util

max_sr=1000
max_points=1000
target_sr = Fs
hop_length = 1

# Pad an extra channel dimension, if necessary
if max_points is not None:

  if max_points < x.shape[-1]:
    target_sr = min(max_sr, (Fs * x.shape[-1]) // max_points)

  hop_length = Fs // target_sr

# Reduce by envelope calculation
# x_frame = np.abs(lib_util.frame(x, frame_length=hop_length, hop_length=hop_length))
x_frame = lib_util.frame(x, frame_length=hop_length, hop_length=hop_length)
max_val = x_frame.max(axis=1)
x_frame.shape


plt.figure(figsize=(6, 1.8))
plt.plot(x_frame.squeeze(), color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()




# librosa to load mp4
# file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/24.03.2022/fa2e5abd-2aab-4d35-a700-56535437b008/evc_API/1038927188.mp4'
file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/24.03.2022/fa2e5abd-2aab-4d35-a700-56535437b008/evc_API/2038926693.mp4'
x, Fs = librosa.load(file_A, sr=None)


sd.play(x, Fs)

sd.stop()

# plt.figure(figsize=(6, 1.8))
# plt.plot(x, color='gray')
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()

new_fs = 8000
y_8k = librosa.resample(x, orig_sr=Fs, res_type='kaiser_fast',target_sr=new_fs)
y_8k.shape
file_downsampled = file_A.replace('.mp4', f'downsampled_{new_fs}.pickle')
import carlos_utils.file_utils as fu
fu.toPickleFile(y_8k, file_downsampled)

# plt.figure(figsize=(6, 1.8))
# plt.plot(y_8k, color='gray')
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()

sd.play(y_8k, new_fs)
sd.stop()

frame_length = int(new_fs*3.5)
hop_length = int(new_fs)
split_signal = librosa.effects.split(y=y_8k,  frame_length=frame_length, hop_length=hop_length, top_db=10)

len(split_signal)

t_axis_seconds = np.arange(y_8k.shape[0])/new_fs
t_axis_mins = t_axis_seconds/60


# import numpy as np
# speech_intervals = np.zeros(shape=len(y_8k),dtype=bool)
# min_interval_secs = 0.5
# for this_interval in split_signal:
#   duration_interval = (this_interval[1]-this_interval[0])/new_fs
#   if duration_interval >= min_interval_secs:
#     speech_intervals[this_interval[0]:this_interval[1]]=True


import numpy as np
speech_intervals = np.zeros(shape=len(y_8k),dtype=bool)
min_interval_secs = 0.5
for this_interval in split_signal:
  duration_interval = (this_interval[1]-this_interval[0])/new_fs
  speech_intervals[this_interval[0]:this_interval[1]]=True

fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))

ax1 = axes[0]
ax1.plot(t_axis_mins, y_8k, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_mins, speech_intervals, color='red')
fig.tight_layout()

plt.show()


# previous_gap = 


## two speakers
# alternatively...measure the power of the signal??
new_fs = 8000

file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/1039006966.mp4'
spk_A, Fs_A = librosa.load(file_A, sr=None)

# The number of samples per analysis frame
frame_length = int(Fs_A*3.5)
# The number of samples between analysis frames
hop_length = int(Fs_A/0.5)
split_signal_A = librosa.effects.split(y=spk_A,  frame_length=frame_length, hop_length=hop_length, top_db=10)
slots_seconds_A = split_signal_A/Fs_A
slots_mins_A = slots_seconds_A/60.0



# for plotting
new_fs = 8000
y_spk_A_8k = librosa.resample(spk_A, orig_sr=Fs_A, res_type='kaiser_fast', target_sr=new_fs)
t_axis_seconds = np.arange(y_spk_A_8k.shape[0])/new_fs




num_samples = len(spk_A)
# np.rint(num_samples/Fs_A, dtype=np.int)
max_seconds = int(0.5+(num_samples/Fs_A))
speech_intervals = np.zeros(shape=max_seconds, dtype=bool)
t_seconds = np.arange(max_seconds)

min_interval_secs = 1
for this_interval in slots_seconds_A.astype(int):
  duration_interval = this_interval[1]-this_interval[0]
  if duration_interval >= min_interval_secs:
    speech_intervals[this_interval[0]:this_interval[1]]=True


fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))

ax1 = axes[0]
ax1.plot(t_axis_seconds, y_spk_A_8k, color='blue')
ax2 = axes[1]
ax2.plot(t_seconds, speech_intervals, color='red')
ax1.grid(True)
ax2.grid(True)
fig.tight_layout()

plt.show()



file_B = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/2039006699.mp4'
spk_B, Fs_B = librosa.load(file_B, sr=None)
split_signal_B = librosa.effects.split(y=spk_B,  frame_length=frame_length, hop_length=hop_length, top_db=10)



speech_intervals_A = np.zeros(shape=len(spk_A),dtype=bool)
min_interval_secs = 0.5
for this_interval in split_signal_A:
  duration_interval = (this_interval[1]-this_interval[0])/Fs_A
  speech_intervals_A[this_interval[0]:this_interval[1]]=True


 

speech_intervals_B = np.zeros(shape=len(spk_B),dtype=bool)
min_interval_secs = 0.5
for this_interval in split_signal_B:
  duration_interval = (this_interval[1]-this_interval[0])/Fs_B
  speech_intervals_B[this_interval[0]:this_interval[1]]=True



t_axis_seconds_A = np.arange(spk_A.shape[0])/Fs_A
t_axis_mins_A = t_axis_seconds_A/60

t_axis_seconds_B = np.arange(spk_B.shape[0])/Fs_B
t_axis_mins_B = t_axis_seconds_B/60

fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))

ax1 = axes[0]
ax1.plot(t_axis_mins_A, speech_intervals_A, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_mins_B, speech_intervals_B, color='red')
fig.tight_layout()

plt.show()



# manually
# The number of samples per window
window_seconds = 0.25
frame_length = int(Fs_A*window_seconds)
# The number of samples between analysis frames
hop_length = int(Fs_A*0.5)
frames = librosa.util.frame(spk_A, frame_length=frame_length, hop_length=hop_length)
# frames.shape 
power = np.mean(np.abs(frames) ** 2, axis=-2, keepdims=True)

plt.plot(power)
plt.show()

# np.power(spk_A, 2)
spk_A_abs = np.abs(spk_A)
# 
spk_max = spk_A_abs.max()
spk_min = spk_A_abs.min()

spk_A_scaled = (spk_A_abs-spk_min)/(spk_max-spk_min)
voice_threshold = 0.2
spk_A_voice = spk_A_scaled >= voice_threshold

plt.plot(spk_A_voice)
plt.show()


from scipy import signal
num_samples = len(spk_A)
max_seconds = int(0.5+(num_samples/Fs_A))
num_samples_second = 4
total_resampled_length = max_seconds*num_samples_second
downsampled =  signal.resample(spk_A_voice.astype(float), total_resampled_length)




'''

/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/26c86a91-ef03-433c-bd12-98880c4e6399b.mp4

The audios are not aligned??
If aligned with the main video
2039006699 starts at 01m02s
1039006966 starts at 01m10s
'''



from scipy import signal, fftpack
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa

new_fs = 8000
file_A = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/26c86a91-ef03-433c-bd12-98880c4e6399b.mp3'
spk_A, Fs_A = librosa.load(file_A, sr=None)
y_spk_A_8k = librosa.resample(spk_A, orig_sr=Fs_A, res_type='kaiser_fast', target_sr=new_fs)

file_B = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/1039006966.mp4'
spk_B, Fs_B = librosa.load(file_B, sr=None)
y_spk_B_8k = librosa.resample(spk_B, orig_sr=Fs_B, res_type='kaiser_fast', target_sr=new_fs)

spk_A = spk_B = None

num_samples = np.min(np.array([y_spk_A_8k.shape[0], y_spk_B_8k.shape[0]])*0.10).astype(int)

A = fftpack.fft(y_spk_A_8k[0:num_samples])
B = fftpack.fft(y_spk_B_8k[0:num_samples])
Ar = -A.conjugate()
Br = -B.conjugate()
xcorr_values = np.abs(fftpack.ifft(Ar*B))

Br = -B.conjugate()
xcorr_values_b = np.abs(fftpack.ifft(A*Br))
idx_max_b = np.argmax(xcorr_values_b)
tau_seconds_b = idx_max_b /new_fs

t_axis_seconds = np.arange(num_samples)/new_fs
t_axis_mins = t_axis_seconds/60

plt.plot(t_axis_seconds, xcorr_values)
plt.show()


idx_max = np.argmax(xcorr_values)
tau_seconds = idx_max/new_fs

plt.plot(t_axis_seconds, y_spk_A_8k[0:num_samples], color='blue', alpha=0.9)
plt.plot(t_axis_seconds+tau_seconds, y_spk_B_8k[0:num_samples], color='red', alpha=0.4)
plt.show()


plt.plot(t_axis_seconds, y_spk_A_8k[0:num_samples], color='blue', alpha=0.9)
plt.plot(t_axis_seconds+tau_seconds_b, y_spk_B_8k[0:num_samples], color='red', alpha=0.4)
plt.show()



from scipy.signal import butter, lfilter
order = 6
cutoff = 1200
nyq = 0.5 * new_fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='high', analog=False)
y_spk_A_8k_filt = lfilter(b, a, y_spk_A_8k)


fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))
axes[0].plot(t_axis_seconds, y_spk_A_8k[0:num_samples], color='blue', alpha=0.9)
axes[1].plot(t_axis_seconds, y_spk_A_8k_filt[0:num_samples], color='blue', alpha=0.9)
plt.show()





######
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
num_intervals = 8
y_spk_A_16k_chunk = y_spk_A_16k[split_signal_A[0][0]:split_signal_A[num_intervals][1]]


split_signal_B = librosa.effects.split(y=y_spk_B_16k,  frame_length=frame_length, hop_length=hop_length, top_db=10)
num_intervals = 8
y_spk_B_16k_chunk = y_spk_B_16k[split_signal_B[0][0]:split_signal_B[num_intervals][1]]


plt.plot(y_spk_A_16k_chunk)
plt.show()


# import time
# sd.play(y_spk_A_16k_chunk, new_fs)
# time.sleep(12)
# sd.stop()


#
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

#Loading the model and the tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

#Tokenize
input_values = tokenizer(y_spk_A_16k_chunk, return_tensors="pt").input_values
#Take logits
logits = model(input_values).logits
#Take argmax
predicted_ids = torch.argmax(logits, dim=-1)
#Get the words from predicted word ids
transcription = tokenizer.decode(predicted_ids[0])


# it's 4GB...
# from transformers import AutoTokenizer, AutoModelForCTC
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-xlarge-ls960-ft")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-xlarge-ls960-ft")

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-base-ls960")
# model = AutoModel.from_pretrained("facebook/hubert-base-ls960")

# from transformers import AutoTokenizer, AutoModelForCTC
# tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

from transformers import Wav2Vec2Processor, HubertForCTC
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


input_values = processor(y_spk_A_16k_chunk, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])



