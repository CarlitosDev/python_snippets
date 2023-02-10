'''

  _estimation.py

'''


from pyrsistent import v
from scipy import signal, fftpack
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import carlos_utils.file_utils as fu
import soundfile as sf


window_in_seconds = 0.5
fixed_threshold = 0.5

baseFolder = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API'
audioFile_A = '1039006966_chunk.wav'
path_to_audio_extract = fu.fullfile(baseFolder, audioFile_A)

path_to_audio_extract='/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/transcriptions/1039006966_transcription_chunk_1.wav'
# path_to_audio_extract='/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/transcriptions/2039006699_transcription_chunk_0.wav'


baseFolder, fName,_ = fu.fileparts(path_to_audio_extract)

spk_A, Fs_A = librosa.load(path_to_audio_extract, sr=None)
num_samples = spk_A.shape[0]


t_axis_seconds = np.arange(num_samples)/Fs_A


window_in_samples = int(window_in_seconds*Fs_A)

conv_mask = np.blackman(window_in_samples)
# plt.plot(conv_mask)
# plt.show()



signal_abs = np.abs(spk_A)
convolved_data = np.convolve(signal_abs, conv_mask, 'same')


fig_h = 10
fig_w = 18
fig, axes = plt.subplots(3,1, figsize=(fig_w, fig_h))
ax1 = axes[0]
ax1.plot(t_axis_seconds, spk_A, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_seconds, signal_abs, color='red')
ax3 = axes[2]
ax3.plot(convolved_data, color='blue')
# ax3.plot(t_axis_seconds, convolved_data, color='red')
plt.show()




random_threshold = np.quantile(convolved_data, fixed_threshold)

speech_windows = convolved_data >= random_threshold

spk_A_windowed = spk_A*speech_windows.astype(float)


fig_h = 10
fig_w = 18
fig, axes = plt.subplots(3,1, figsize=(fig_w, fig_h))

ax1 = axes[0]
ax1.plot(t_axis_seconds, spk_A, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_seconds, speech_windows, color='red')
ax3 = axes[2]
ax3.plot(t_axis_seconds, spk_A_windowed, color='green')

fig.tight_layout()
fig_params = f'window={window_in_seconds},threshold={fixed_threshold:3.2f}'


pfg_file_name = fu.fullfile(baseFolder, f'{fName}_{fig_params}.png')
plt.savefig(pfg_file_name)







# convolve with reflection to smooth edges out
# window_in_seconds = 0.1


conv_mask = np.blackman(window_in_samples)
padding_in_samples = int(window_in_samples)
# signal_abs_padded = np.pad(signal_abs, (padding_in_samples, padding_in_samples), 'reflect', 'reflect_type'='odd')
# signal_abs_padded = np.pad(signal_abs, (padding_in_samples, padding_in_samples), 'wrap')
signal_abs_padded = np.pad(signal_abs, (0, padding_in_samples), 'reflect')
# signal_abs_padded = np.concatenate([np.zeros(padding_in_samples), signal_abs, np.zeros(padding_in_samples)], axis=0)
# signal_abs_padded = np.concatenate([signal_abs, np.zeros(padding_in_samples)], axis=0)
signal_abs_padded = np.concatenate([np.pad(signal_abs, (0, padding_in_samples), 'reflect'), np.zeros(padding_in_samples)], axis=0)
# plt.plot(signal_abs_padded, color='blue')
# plt.show()
convolved_data = np.convolve(signal_abs_padded, conv_mask, 'valid')[:-1]
convolved_data.shape


mn = convolved_data.min()
mx = convolved_data.max()
convolved_data_norm = (convolved_data-mn)/(mx-mn)
# convolved_data_norm = convolved_data_norm[padding_in_samples:-padding_in_samples]
# convolved_data_norm = convolved_data_norm[padding_in_samples::]
convolved_data_norm = convolved_data_norm[:-padding_in_samples]
convolved_data_norm.shape



fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))
ax1 = axes[0]
ax1.plot(t_axis_seconds, spk_A, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_seconds, signal_abs, color='red', alpha=0.5)
ax2.plot(t_axis_seconds, convolved_data_norm, color='green')
# ax2.plot(convolved_data_norm_b, color='green')
plt.show()





########

window_in_seconds = 0.1
window_in_samples = int(window_in_seconds*Fs_A)
conv_mask = np.blackman(window_in_samples)
padding_in_samples = int(window_in_samples)
signal_abs = np.abs(spk_A)
signal_abs_padded = np.pad(signal_abs, (padding_in_samples, padding_in_samples), 'reflect')
convolved_data = np.convolve(signal_abs_padded, conv_mask, 'valid')[:-1]
convolved_data.shape


mn = convolved_data.min()
mx = convolved_data.max()
convolved_data_norm = (convolved_data-mn)/(mx-mn)
# convolved_data_norm = convolved_data_norm[padding_in_samples:-padding_in_samples]
# convolved_data_norm = convolved_data_norm[padding_in_samples::]
edges_to_remove = int(padding_in_samples/2)
# convolved_data_norm = convolved_data_norm[:-padding_in_samples]
convolved_data_norm = convolved_data_norm[edges_to_remove:-edges_to_remove]
convolved_data_norm.shape



fig_h = 10
fig_w = 18
fig, axes = plt.subplots(2,1, figsize=(fig_w, fig_h))
ax1 = axes[0]
ax1.plot(t_axis_seconds, spk_A, color='blue')
ax2 = axes[1]
ax2.plot(t_axis_seconds, signal_abs, color='red', alpha=0.5)
ax2.plot(t_axis_seconds, convolved_data_norm, color='green')
# ax2.plot(convolved_data_norm_b, color='green')
plt.show()

