
'''

source ~/.bash_profile && AWS_PROFILE=efdata-qa && python3


Requisites:
	pip3 install pytube
'''


import os
from pytube import YouTube
import carlos_utils.file_utils as fu
resolution = 720
resolution_str = f'{resolution}p'

path_to_videos = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/DataScience/NLP/CMU-course'

links = ['https://www.youtube.com/watch?v=rVht4eK3EZw', 'https://www.youtube.com/watch?v=boPpVexvDAI', 
'https://www.youtube.com/watch?v=pifqfW2ApI4', 'https://www.youtube.com/watch?v=N_Ip2zhIGSk', 
'https://www.youtube.com/watch?v=FazNgBWvkkk', 'https://www.youtube.com/watch?v=0PPzD4mxpuM', 
'https://www.youtube.com/watch?v=27LkyrxaUK4', 
'https://www.youtube.com/watch?v=BXPyIENMs4Y', 'https://www.youtube.com/watch?v=5ef83Wljm-M']

# for _link in links:
for _link in []:


	print(f'Downloading {_link}')

	yt = YouTube(_link)

	title = yt.title
	description = yt.description
	mp4_video = yt.streams.filter(
		file_extension="mp4").get_by_resolution(resolution_str)

	file_name = title.replace(':', '_') + '.mp4'

	outpath = os.path.join(path_to_videos, file_name)
	try:
		if not os.path.exists(outpath):
			mp4_video.download(outpath)

		caption = yt.captions['a.en']
		xml_captions = caption.xml_captions
		json_captions = caption.json_captions

		json_path = outpath.replace('.mp4', '.json')

		fu.writeJSONFile(json_captions, json_path)
	except Exception as ex:
		print(f'Something went wrong {ex}')



import os
import time
import EVC_utils.lesson_analysis_settings as lse
import glob
from EVC_API.evc_detect_lesson_parts import get_lesson_parts
import carlos_utils.video_utils_scene_detection as vsd

analysis_settings = lse.get_analysis_settings()
video_settings = analysis_settings['video_analysis']
DETECTOR_TYPE = video_settings['detector_type']
DETECTOR_THRESHOLD = video_settings['detector_threshold']


# instanciate the OCR tool
from  carlos_utils.OCR_image_utils import get_paddleOCR_engine
ocr_engine = get_paddleOCR_engine(language='en')

rootFolder = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/DataScience/NLP/CMU-course/'
glob_pattern = os.path.join(rootFolder, '*', '*.mp4')
video_lesson_folders = glob.glob(glob_pattern)


shrink_number_scenes = True
total_lessons = len(video_lesson_folders)
for idx_lesson, videopath in enumerate(video_lesson_folders):

	lesson_folder, _, _ = fu.fileparts(videopath)
	scenes_ocr_folder = fu.fullfile(lesson_folder, 'scene_detection', 'ocr')
	fu.makeFolder(scenes_ocr_folder)

	scenes_data_folder = fu.fullfile(lesson_folder, 'scene_detection', 'data')
	fu.makeFolder(scenes_data_folder)
	scene_slots_file = fu.fullfile(
	    scenes_data_folder, 'scene_slots_main_video.pickle')

	# Analyse the video (reload existing analysis)
	scene_list_file = fu.fullfile(scenes_data_folder, 'scene_list.pickle')
	df_metrics_file = fu.fullfile(scenes_data_folder, 'df_metrics.pickle')
	selected_frames_file = fu.fullfile(
	    scenes_data_folder, 'selected_frames.pickle')

	if not fu.fileexists(scene_slots_file):

		try:
			print(f'Processing {idx_lesson}/{total_lessons} - {videopath}...')
			start_time = time.perf_counter()

			scene_list, df_metrics, selected_frames = vsd.process_video_scenes(videopath, DETECTOR_THRESHOLD, DETECTOR_TYPE, reduce_number_detected_scenes=shrink_number_scenes)
      # save the info
			fu.toPickleFile(scene_list, scene_list_file)
			fu.toPickleFile(df_metrics, df_metrics_file)
			fu.toPickleFile(selected_frames, selected_frames_file)

			runTime = time.perf_counter() - start_time
			print(f'Video processed in {runTime:.2f} seconds')
		except Exception as ex:
			print(f'Something went wrong {ex}')