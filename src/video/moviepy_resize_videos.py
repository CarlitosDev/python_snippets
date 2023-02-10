'''
moviepy_resize-videos.py
'''


import os
import time
import utils.file_utils as fu
import utils.utils_root as ur
import utils.aws_data_utils as awsu
from joblib import Parallel, delayed
import moviepy.editor as mp

video_extension = 'mp4'


base_folder = '/Users/carlos.aguilar/Documents/EF_Content/EFxAWS/'
videos_folder = fu.fullfile(base_folder, 'Unilever_videos')

video_list = os.listdir(videos_folder)
path_to_videofiles = [videos_folder + '/' + iFile \
   for iFile in video_list if video_extension in iFile]




path_to_video = path_to_videofiles[1]


def resize_video(path_to_video):
  clip = mp.VideoFileClip(path_to_video)
  fPath, file, ext = fu.fileparts(path_to_video)
  path_to_resized_video = os.path.join(fPath, 'resized', file + \
    ext.replace('.mp4', '_resized.mp4'))
  sb_clip = clip.resize(0.25)
  sb_clip.write_videofile(path_to_resized_video, codec='libx264')


def parallel_video_caller(path_to_videofiles):
  for this_video in path_to_videofiles:
    resize_video(this_video)



num_processes = 6
num_files_to_process = len(path_to_videofiles)
num_files = int(0.5+(num_files_to_process/num_processes))

videoList = [path_to_videofiles[i:i + num_files] \
  for i in range(0, len(path_to_videofiles), num_files)]

queryStart = time.time()
r = Parallel(n_jobs=-1)(delayed(parallel_video_caller)(file_selection) \
  for file_selection in videoList)
queryElapsed = time.time() - queryStart
print(f'...arranging videos done in {queryElapsed:.2f} sec!')