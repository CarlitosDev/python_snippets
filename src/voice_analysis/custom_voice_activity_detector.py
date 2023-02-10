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


# read the audio file
baseFolder = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API'
audioFile_A = '1039006966_chunk.wav'
path_to_audio_extract = fu.fullfile(baseFolder, audioFile_A)

path_to_audio_extract='/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/transcriptions/1039006966_transcription_chunk_1.wav'
# path_to_audio_extract='/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/26c86a91-ef03-433c-bd12-98880c4e6399/evc_API/transcriptions/2039006699_transcription_chunk_0.wav'


baseFolder, fName,_ = fu.fileparts(path_to_audio_extract)

spk_A, Fs_A = librosa.load(path_to_audio_extract, sr=None)




def detect_voice(spk_A, Fs_A, window_in_seconds=0.1, voice_min_amplitude=0.15, plot_results=False):

  num_samples = spk_A.shape[0]

  window_in_samples = int(window_in_seconds*Fs_A)
  conv_mask = np.blackman(window_in_samples)
  padding_in_samples = int(window_in_samples)
  signal_abs = np.abs(spk_A)
  signal_abs_padded = np.pad(signal_abs, (padding_in_samples, padding_in_samples), 'reflect')
  convolved_data = np.convolve(signal_abs_padded, conv_mask, 'valid')[:-1]

  mn = convolved_data.min()
  mx = convolved_data.max()
  convolved_data_norm = (convolved_data-mn)/(mx-mn)
  edges_to_remove = int(padding_in_samples/2)
  convolved_data_norm = convolved_data_norm[edges_to_remove:-edges_to_remove]

  speech_windows = convolved_data_norm >= voice_min_amplitude
  spk_A_windowed = spk_A*speech_windows.astype(float)

  if plot_results:
    fig_h = 10
    fig_w = 18
    t_axis_seconds = np.arange(num_samples)/Fs_A
    fig, axes = plt.subplots(3,1, figsize=(fig_w, fig_h))
    axes[0].plot(t_axis_seconds, spk_A, color='blue')
    axes[1].plot(t_axis_seconds, signal_abs, color='red', alpha=0.5)
    axes[1].plot(t_axis_seconds, convolved_data_norm, color='green')
    axes[2].plot(t_axis_seconds, spk_A_windowed, color='green')
    plt.show()

  return spk_A_windowed, speech_windows








spk_A_windowed, speech_windows = detect_voice(spk_A, Fs_A, window_in_seconds= 0.2, voice_min_amplitude= 0.15, plot_results=True)
speech_slots, speech_slots_seconds = get_speech_slots(speech_windows, Fs_A)


def get_speech_slots(speech_windows, Fs_A):

  vector_idx = np.arange(speech_windows.shape[0])
  # get the 1st order difference
  derivative = np.diff(speech_windows.astype(float), n=1,prepend=[0])

  slot_begins = vector_idx[derivative==1]
  slot_ends   = vector_idx[derivative==-1]

  # sanity checks to find closed slots
  if slot_ends[0] < slot_begins[0]:
    slot_ends = np.delete(slot_ends, 0)

  if slot_ends[-1] < slot_begins[-1]:
    slot_begins = np.delete(slot_begins, -1)

  num_slots_starts = len(slot_begins)
  num_slots_ends = len(slot_ends)

  if num_slots_starts == num_slots_ends:
    speech_slots = np.column_stack([slot_begins, slot_ends])
  else:
    total_slots = np.min([num_slots_starts,num_slots_ends])
    speech_slots = np.column_stack([slot_begins[:total_slots], slot_ends[:total_slots]])

  speech_slots_seconds = speech_slots/Fs_A
  speech_slots_duration_seconds = speech_slots_seconds[:,1]-speech_slots_seconds[:,0]
  
  return speech_slots, speech_slots_seconds, speech_slots_duration_seconds