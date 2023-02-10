'''

	librosa_downsampling_comparison.py

'''

import os
import carlos_utils.audio_utils as audu
import soundfile as sf
import carlos_utils.file_utils as fu

baseFolder = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/28.03.2022/fdd9a173-3595-4752-874c-a2ff27dea46c/evc_API/'

path_to_signal = os.path.join(baseFolder, 'fdd9a173-3595-4752-874c-a2ff27dea46cb.mp4')
file_A = '1039006944.mp4'
path_to_signal_A = os.path.join(baseFolder, file_A)
file_B = '2039006716.mp4'
path_to_signal_B = os.path.join(baseFolder, file_B)


new_fs = 4000
# Process 'master' track
online_lesson_audio, Fs = audu.read_audio_file(path_to_signal)
audio_downsampled_1 = audu.resample_audio(online_lesson_audio, Fs, new_fs, sampling_method='fft')
audio_downsampled_2 = audu.resample_audio(online_lesson_audio, Fs, new_fs, sampling_method='kaiser_fast')
audio_downsampled_3 = audu.resample_audio(online_lesson_audio, Fs, new_fs, sampling_method='linear')
audio_downsampled_4 = audu.resample_audio(online_lesson_audio, Fs, new_fs, sampling_method='polyphase')
'''
Finished 'read_audio_file' in 1.6727 secs
Finished 'resample_audio' in 2.5956 secs
Finished 'resample_audio' in 13.2315 secs
Finished 'resample_audio' in 0.1433 secs
Finished 'resample_audio' in 2.0592 secs
'''

participant_A_audio, Fs_A = audu.read_audio_file(path_to_signal_A)
participant_A_audio_downsampled_1 = audu.resample_audio(participant_A_audio, Fs_A, new_fs, sampling_method='fft')
participant_A_audio_downsampled_2 = audu.resample_audio(participant_A_audio, Fs_A, new_fs, sampling_method='kaiser_fast')
participant_A_audio_downsampled_3 = audu.resample_audio(participant_A_audio, Fs_A, new_fs, sampling_method='linear')
participant_A_audio_downsampled_4 = audu.resample_audio(participant_A_audio, Fs_A, new_fs, sampling_method='polyphase')
'''
Finished 'read_audio_file' in 2.4353 secs
Finished 'resample_audio' in 48.0079 secs
Finished 'resample_audio' in 18.9087 secs
Finished 'resample_audio' in 0.1740 secs
Finished 'resample_audio' in 3.5760 secs
'''


participant_B_audio, Fs_B = audu.read_audio_file(path_to_signal_B)
participant_B_audio_downsampled_1 = audu.resample_audio(participant_B_audio, Fs_B, new_fs, sampling_method='fft')
participant_B_audio_downsampled_2 = audu.resample_audio(participant_B_audio, Fs_B, new_fs, sampling_method='kaiser_fast')
participant_B_audio_downsampled_3 = audu.resample_audio(participant_B_audio, Fs_B, new_fs, sampling_method='linear')
participant_B_audio_downsampled_4 = audu.resample_audio(participant_B_audio, Fs_B, new_fs, sampling_method='polyphase')
'''
Finished 'read_audio_file' in 2.6537 secs
Finished 'resample_audio' in 97.0599 secs
Finished 'resample_audio' in 19.3061 secs
Finished 'resample_audio' in 0.1813 secs
Finished 'resample_audio' in 3.0105 secs
'''