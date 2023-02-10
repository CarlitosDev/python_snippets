'''
  Create a wav file that contains a speaker on the left and another on the right

'''


import numpy as np
from scipy.io.wavfile import write
samplerate = 44100; fs = 440
t = np.linspace(0., 1., samplerate*5)
amplitude = np.iinfo(np.int16).max
data_left = amplitude * np.sin(2. * np.pi * fs * t)
data_right = amplitude*0.6 * np.sin(2. * np.pi * fs * t)
audio_data = np.array([data_left.astype(np.int16), data_right.astype(np.int16)]).T
# audio_data = np.vstack((data.astype(np.int16),data.astype(np.int16))).T
audio_data.shape
output_file = '/Users/carlos.aguilar/Documents/tempRubbish/example.wav'
write(output_file, samplerate, audio_data)



# Flac is the new wav 
import soundfile as sf
output_file = '/Users/carlos.aguilar/Documents/tempRubbish/example.flac'
sf.write(output_file, audio_data, samplerate, format='flac')




import carlos_utils.video_utils as vu
path_to_video = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/23.06.2022/4f95baf8-46fb-4b2a-850a-173eacb5b2a9/evc_API/4f95baf8-46fb-4b2a-850a-173eacb5b2a9b.mp4'
vu.convert_mp4_video_to_mp3(path_to_video)

path_to_flac_audio = path_to_video.replace('.mp4', '.flac')

st = subprocess.call(["ffmpeg", "-y", "-i", path_to_video, path_to_flac_audio],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)


##
import os
import EVC_utils.identify_speakers as evcaud
from EVC_API.evc_data_types import EVCLesson
import numpy as np
import carlos_utils.file_utils as fu
import soundfile as sf

new_fs = 8000

get_min = lambda a,b: a if a<=b else b

baseFolder = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/23.06.2022/4f95baf8-46fb-4b2a-850a-173eacb5b2a9/evc_API'
main_video_file = '4f95baf8-46fb-4b2a-850a-173eacb5b2a9b.mp4'

path_to_lesson_info = os.path.join(baseFolder, 'API_lesson_info(raw).pickle')
lesson_info = EVCLesson(fu.readPickleFile(path_to_lesson_info))


path_to_signal = os.path.join(baseFolder, main_video_file)
master_audio_downsampled = evcaud.read_and_resample_participant_audio(path_to_signal, new_fs=new_fs)

# process the teacher
audio_teacher = np.zeros_like(master_audio_downsampled)
audio_teacher_samples = audio_teacher.shape[0]

teacher_info = lesson_info.get_host()
teacher_videofile = lesson_info.get_videofile_from_attendance_ref_code(teacher_info.attendance_ref_code)
path_teacher_videofile = fu.fullfile(baseFolder, teacher_videofile)

participant_audio_processed, signal_length_samples, lag_seconds, lag_samples = \
  evcaud.process_audio_signals(path_teacher_videofile, master_audio_downsampled, \
    new_fs=new_fs, do_noisereduction=False)


#
from matplotlib import pyplot as plt
import librosa.display
librosa.display.waveplot(participant_audio_processed, sr=new_fs, color='blue')
plt.show()
#

start_sample = lag_samples

if lag_samples+signal_length_samples < audio_teacher_samples:
  end_sample = lag_samples+signal_length_samples
  audio_teacher[start_sample:end_sample] = participant_audio_processed
else:
  end_sample_audio_teacher = audio_teacher_samples
  end_sample_participant = audio_teacher_samples-lag_samples
  audio_teacher[start_sample:end_sample_audio_teacher] = participant_audio_processed[0:end_sample_participant]


#
librosa.display.waveplot(audio_teacher, sr=new_fs, color='blue')
plt.show()
#

# process the student
audio_student = np.zeros_like(master_audio_downsampled)
audio_student_samples = audio_student.shape[0]

student_info = lesson_info.get_students()[0]
student_videofile = lesson_info.get_videofile_from_attendance_ref_code(student_info.attendance_ref_code)
path_student_videofile = fu.fullfile(baseFolder, student_videofile)

st_audio_processed, st_signal_length_samples, st_lag_seconds, st_lag_samples = \
  evcaud.process_audio_signals(path_student_videofile, master_audio_downsampled, \
    new_fs=new_fs, do_noisereduction=False)

#
from matplotlib import pyplot as plt
import librosa.display
librosa.display.waveplot(st_audio_processed, sr=new_fs, color='blue')
plt.show()
#


st_start_sample = st_lag_samples

if st_start_sample+st_signal_length_samples < audio_student_samples:
  end_sample = st_start_sample+st_signal_length_samples
  audio_student[st_start_sample:end_sample] = st_audio_processed
else:
  end_sample_audio_student = audio_student_samples
  end_sample_participant = audio_student_samples-st_start_sample
  audio_student[st_start_sample:end_sample_audio_student] = st_audio_processed[0:end_sample_participant]


# audio_data = np.array([audio_teacher.astype(np.int16), audio_student.astype(np.int16)]).T
audio_data = np.array([audio_teacher, audio_student]).T

audio_data.shape

librosa.display.waveplot(audio_data.T, sr=new_fs, color='green')
plt.show()

speakers_audio_file = path_to_signal.replace('.mp4', '.flac')
sf.write(speakers_audio_file, audio_data, new_fs, format='flac')


# speakers_audio_file = path_to_signal.replace('.mp4', '.wav')
# sf.write(speakers_audio_file, audio_data, new_fs, format='wav')

import carlos_utils.aws_data_utils as awsu
import EVC_utils.lesson_commander as lecom

input_bucket = 'ef-data-evc-videos-2'
output_bucket = 'ef-data-evc-videos-analysis'
skip_if_exists = True

existing_transcribe_jobs = []
evcAPIFolder, job_id, _ = fu.fileparts(speakers_audio_file)


if skip_if_exists:
  existing_transcribe_jobs = awsu.get_list_of_Transcribe_jobs()

# awsu.delete_list_of_Transcribe_jobs([job_id])
  
if not job_id in existing_transcribe_jobs:
  
  # 2 - upload it to AWS S3
  audio_file = speakers_audio_file.split('/')[-1]
  awsu.upload_file_to_S3(speakers_audio_file, audio_file, input_bucket)

  # 3 - send to AWS Transcribe
  aws_transcribe_english_codes = ['en-US','zh-CN','zh-TW']
  job_settings = {'ChannelIdentification': False}

  job_id = awsu.generic_aws_transcribe_job(audio_file, input_bucket, output_bucket,
    job_settings, language_options=aws_transcribe_english_codes,
    region='eu-west-1')

# wait here
expected_transcribe_path = fu.fullfile(evcAPIFolder, f'{job_id}.json')
lecom.get_aws_transcription(job_id, expected_transcribe_path)