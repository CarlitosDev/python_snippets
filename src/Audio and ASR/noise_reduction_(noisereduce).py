'''
noise_reduction_(noisereduce).py

pip3 install noisereduce

from https://github.com/timsainb/noisereduce

13.05.2022 - This is really good!!
'''



import os
import carlos_utils.file_utils as fu
import carlos_utils.audio_utils as audu
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import noisereduce as nr

# base_folder   = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos'
base_folder   = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons_api'
lesson_subfolder = 'adults_spaces/12.04.2022/3fbd6a03-e10f-4a3e-afaa-4a401556e20d'
lesson_folder = fu.fullfile(base_folder, lesson_subfolder)

# 1 - Read the main audiodile (mp3)
main_audio_file = fu.fullfile(lesson_folder, 'evc_API', '3fbd6a03-e10f-4a3e-afaa-4a401556e20db.mp3')
main_audio, main_audio_fs = audu.read_audio_file(main_audio_file)


# 2 - Read the individual videos and extract the audio tracks.
student_audio_file = fu.fullfile(lesson_folder, 'evc_API', '1032594586.mp4')
student_audio, student_audio_fs = audu.read_audio_file(student_audio_file)
new_fs = 16000
student_audio_downsampled = audu.resample_audio(student_audio, student_audio_fs, new_fs, sampling_method='polyphase')


teacher_audio_file = fu.fullfile(lesson_folder, 'evc_API', '2032594451.mp4')
teacher_audio, teacher_audio_fs = audu.read_audio_file(teacher_audio_file)


# reduced_noise = nr.reduce_noise(y=student_audio, sr=student_audio_fs)
# this flies with 8kHz...A bit slower with 16kHz
reduced_noise = nr.reduce_noise(y=student_audio_downsampled, sr=new_fs)


import soundfile as sf
file_D = fu.fullfile(lesson_folder, 'student_reduced_noise.wav')
sf.write(file_D, reduced_noise, new_fs)