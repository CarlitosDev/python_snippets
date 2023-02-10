'''
scene_detection.py


pip3 install scenedetect

pip3 install scenedetect --upgrade


cd "/Users/carlos.aguilar/Documents/EF_EVC_API_videos/adults_spaces/16.06.2022/02dfb48d-97e5-4592-a160-5c9c00a86fb8/PySceneDetection"
scenedetect --input ./../evc_API/02dfb48d-97e5-4592-a160-5c9c00a86fb8b.mp4 --stats 02dfb48d-97e5-4592-a160-5c9c00a86fb8b.stats.csv detect-content split-video

cd "/Users/carlos.aguilar/Documents/EF_EVC_API_videos/adults_spaces/15.06.2022/fb073b7a-cdd5-4d01-ba34-99bc95803ea9/PySceneDetection"

scenedetect --input ./../evc_API/2033517452.mp4 --stats 2033517452.stats.csv detect-content split-video

scenedetect --input ./../evc_API/2033517452.mp4 --stats 2033517452.stats.csv detect-content list-scenes save-images

cd "/Users/carlos.aguilar/Documents/EF_EVC_API_videos/adults_spaces/15.06.2022/fb073b7a-cdd5-4d01-ba34-99bc95803ea9/PySceneDetection_adaptive"
scenedetect --input ./../evc_API/2033517452.mp4 --stats 2033517452.stats.csv detect-adaptive list-scenes save-images


'''

from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector


def find_scenes_2(video_path):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        # Each scene is a tuple of (start, end) FrameTimecodes.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print(
                'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            base_timecode = video_manager.get_base_timecode()
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return scene_list


path_to_video = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videolessons/20.02.2022/6e04e0d8-3804-45dd-a553-ad4917eebab8/6e04e0d8-3804-45dd-a553-ad4917eebab8.mp4'
scenes = find_scenes(path_to_video)
print(scenes)

scenes = find_scenes_2(path_to_video)



#####
## How to detect scenes of a video and extract the images so we can pass them onto Yolo, 
# or a custom SIFT detector, etc
#
# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager
# For content-aware scene detection:

# Detects fast cuts using changes in colour and intensity between frames.
from scenedetect.detectors.content_detector import ContentDetector

from scenedetect.frame_timecode import FrameTimecode
import carlos_utils.image_utils as imgu
import matplotlib.pyplot as plt


video_path = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/13.07.2022/d0ed81d2-74bc-4591-abf5-707eb8b7c7c9/evc_API/d0ed81d2-74bc-4591-abf5-707eb8b7c7c9b.mp4'
video_path = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/29.06.2022/bb2337dd-78f0-4f61-bf32-8c5a06711c97/evc_API/bb2337dd-78f0-4f61-bf32-8c5a06711c97b.mp4'
video_path = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/29.06.2022/bb2337dd-78f0-4f61-bf32-8c5a06711c97/evc_API/2033712678.mp4'

video_manager = VideoManager([video_path])
stats_manager = StatsManager()
# Construct our SceneManager and pass it our StatsManager.
scene_manager = SceneManager(stats_manager)


# Add ContentDetector algorithm (each detector's constructor
# takes detector options, e.g. threshold).
threshold = 17
scene_manager.add_detector(ContentDetector(threshold=threshold))

# Set downscale factor to improve processing speed.
# video_manager.set_downscale_factor()

# Perform scene detection on video_manager.
scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)


# Obtain list of detected scenes. Each scene is a tuple of (start, end) FrameTimecodes.
scene_list = scene_manager.get_scene_list()


for i, scene in enumerate(scene_list):
    start_time = scene[0].get_timecode()
    end_time = scene[1].get_timecode()
    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()
    total_frames = end_frame - start_frame
    total_seconds = total_frames/video_manager.frame_rate
    print(f'Scene {i:2d}: {total_seconds:.1f} seconds. From {start_time} to {end_time}; {total_frames} frames ({start_frame},{end_frame})''')

base_timecode = video_manager.get_base_timecode()
# stats_manager.save_to_csv(stats_file, base_timecode)

video_manager.reset()



idx_scene = 2
detected_scene = scene_list[idx_scene]
scene_start_frame = detected_scene[0].get_frames()
scene_end_frame = detected_scene[1].get_frames()
middle_frame = round((scene_start_frame+scene_end_frame)/2)

middle_frame_timecode = FrameTimecode(middle_frame, video_manager.frame_rate)


video_manager.seek(middle_frame_timecode)


frame_im = imgu.convert_img_from_BGR_to_RGB(video_manager.read())
imgu.show_image(frame_im)




fig, ax = plt.subplots()
for idx_scene,detected_scene in enumerate(scene_list):
    detected_scene = scene_list[idx_scene]
    scene_start_frame = detected_scene[0].get_frames()
    scene_end_frame = detected_scene[1].get_frames()
    middle_frame = round((scene_start_frame+scene_end_frame)/2)

    middle_frame_timecode = FrameTimecode(middle_frame, video_manager.frame_rate)


    video_manager.seek(middle_frame_timecode)

    frame_im = imgu.convert_img_from_BGR_to_RGB(video_manager.read())
    im = ax.imshow(frame_im)
    ax.set_title(f'Scene {idx_scene}')
    ax.axis('off')
    plt.pause(1)
plt.show()



# for i, scene_timecodes in enumerate(timecode_list):
#     for j, image_timecode in enumerate(scene_timecodes):
#         video.seek(image_timecode)
#         frame_im = video.read()

# stats_manager.save_to_csv(stats_file, base_timecode)
# Adapt the code from stats_manager to arrange a DF
metric_keys = sorted(list(stats_manager._registered_metrics.union(stats_manager._loaded_metrics)))
COLUMN_NAME_FRAME_NUMBER = "Frame Number"
COLUMN_NAME_TIMECODE = "Timecode"
column_names = [COLUMN_NAME_FRAME_NUMBER, COLUMN_NAME_TIMECODE] + metric_keys

frame_keys = sorted(stats_manager._frame_metrics.keys())
metrics = []
for frame_key in frame_keys:
    frame_timecode = stats_manager._base_timecode + frame_key
    metrics.append(
    [frame_timecode.get_frames() +1, frame_timecode.get_timecode()] + \
    [str(metric) for metric in stats_manager.get_metrics(frame_key, metric_keys)]
    )

import pandas as pd
# dtypes=list(zip(column_names,[int, str, float, float, float, float]))
# dtypes=np.dtype(list(zip(column_names,[int, str, float, float, float, float])))
# df_metrics = pd.DataFrame(metrics, columns=column_names, dtype=dtypes)

dtypes=dict(zip(column_names,[int, str, float, float, float, float]))
df_metrics = pd.DataFrame(metrics, columns=column_names)
for column in df_metrics.columns:
    df_metrics[column] = df_metrics[column].astype(dtypes[column])
df_metrics.dtypes

import carlos_utils.plot_utils as pltu
pltu.plot(df_metrics['Frame Number'], df_metrics['content_val'])



import plotly.express as px
fig = px.line(x=df_metrics['Frame Number'], y=df_metrics['content_val'])
fig.show()



###

frame_values = dict(zip(df_metrics['Frame Number'].values, df_metrics['content_val'].values))

# alternatively show all the detected images
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for idx_scene,detected_scene in enumerate(scene_list):
    detected_scene = scene_list[idx_scene]
    scene_start_frame = detected_scene[0].get_frames()
    scene_end_frame = detected_scene[1].get_frames()
    middle_frame = round((scene_start_frame+scene_end_frame)/2)

    middle_frame_timecode = FrameTimecode(middle_frame, video_manager.frame_rate)


    video_manager.seek(middle_frame_timecode)

    frame_im = imgu.convert_img_from_BGR_to_RGB(video_manager.read())
    im = ax.imshow(frame_im)
    content_val = frame_values.get(scene_start_frame+1, -1)
    ax.set_title(f'Scene {idx_scene} ({content_val:.2f})')
    ax.axis('off')
    plt.pause(2)
plt.show()




## Save the images
import carlos_utils.file_utils as fu
scenes_folder = fu.fullfile(fu.fileparts(video_path)[0], '..', 'scene_detection')
fu.makeFolder(scenes_folder)

frame_values = dict(zip(df_metrics['Frame Number'].values, df_metrics['content_val'].values))
video_manager.reset()

fig, ax = plt.subplots()
for idx_scene, detected_scene in enumerate(scene_list):

    start_time = detected_scene[0].get_timecode()
    end_time = detected_scene[1].get_timecode()
    start_frame = detected_scene[0].get_frames()
    end_frame = detected_scene[1].get_frames()
    
    middle_frame = round((start_frame+end_frame)/2)

    middle_frame_timecode = FrameTimecode(middle_frame, video_manager.frame_rate)

    total_frames = end_frame - start_frame
    total_seconds = total_frames/video_manager.frame_rate
    scene_details = f'Scene {idx_scene:2d} - {total_seconds:.1f} seconds. From {start_time} to {end_time} - {total_frames} frames ({start_frame},{end_frame})'''
    print(scene_details)

    video_manager.seek(middle_frame_timecode)

    frame_im = imgu.convert_img_from_BGR_to_RGB(video_manager.read())
    im = ax.imshow(frame_im)
    content_val = frame_values.get(start_frame+1, -1)
    ax.set_title(f'Scene {idx_scene} ({content_val:.2f})')
    ax.axis('off')
    fig_path = fu.fullfile(scenes_folder, scene_details.replace(':', '.') + '.png')
    plt.tight_layout()
    plt.savefig(fig_path)




## Same as above but include object detection
import torch
import glob
import os
import matplotlib.pyplot as plt
yolo_model='yolov5m'
show_detection=True
yolo_model = torch.hub.load('ultralytics/yolov5', yolo_model, pretrained=True)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


base_folder = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons_api/adults_spaces/13.07.2022/d0ed81d2-74bc-4591-abf5-707eb8b7c7c9/scene_detection'
glob_pattern = os.path.join(base_folder, '*.png')
png_files = glob.glob(glob_pattern)


# this_file_path = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/29.06.2022/bb2337dd-78f0-4f61-bf32-8c5a06711c97/scene_detection (teacher video)/Scene  7 - 23.5 seconds. From 00.13.44.333 to 00.14.07.800 - 352 frames (12365,12717).png'
this_file_path = png_files[2]

this_image = imgu.load_image(this_file_path)
imgu.show_image(this_image)

fig, ax = plt.subplots()
im = ax.imshow(this_image)
ax.axis('off')
plt.show()


import cv2
image = cv2.imread(this_file_path)
results = yolo_model(image)

results = yolo_model(this_image)
# results.print()

# if show_detection:
# results.show()

# df_results = results.pandas().xyxy[0]
# _cols = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence']

# detected_objects = {}
# for _, iRow in df_results.iterrows():
# detected_objects.setdefault(iRow['name'], []).append(iRow[_cols].to_dict())




# import carlos_utils.computer_vision_utils as cvis
# detected_objects = cvis.object_detection_yolov5(this_image, yolo_model='yolov5m', show_detection=True)
# fu.printJSON(detected_objects)


import carlos_utils.file_utils as fu
scenes_folder = fu.fullfile(fu.fileparts(video_path)[0], '..', 'scene_detection')
fu.makeFolder(scenes_folder)

frame_values = dict(zip(df_metrics['Frame Number'].values, df_metrics['content_val'].values))
video_manager.reset()

fig, ax = plt.subplots()
for idx_scene, detected_scene in enumerate(scene_list):

    start_time = detected_scene[0].get_timecode()
    end_time = detected_scene[1].get_timecode()
    start_frame = detected_scene[0].get_frames()
    end_frame = detected_scene[1].get_frames()
    
    middle_frame = round((start_frame+end_frame)/2)

    middle_frame_timecode = FrameTimecode(middle_frame, video_manager.frame_rate)

    total_frames = end_frame - start_frame
    total_seconds = total_frames/video_manager.frame_rate
    scene_details = f'Scene {idx_scene:2d} - {total_seconds:.1f} seconds. From {start_time} to {end_time} - {total_frames} frames ({start_frame},{end_frame})'''
    print(scene_details)

    video_manager.seek(middle_frame_timecode)

    frame_im = imgu.convert_img_from_BGR_to_RGB(video_manager.read())
    im = ax.imshow(frame_im)
    content_val = frame_values.get(start_frame+1, -1)
    ax.set_title(f'Scene {idx_scene} ({content_val:.2f})')
    ax.axis('off')
    fig_path = fu.fullfile(scenes_folder, scene_details.replace(':', '.') + '.png')
    plt.tight_layout()
    plt.savefig(fig_path)