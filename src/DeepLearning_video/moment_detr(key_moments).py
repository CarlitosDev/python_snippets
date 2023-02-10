'''


	The moment_detr allows to search for text in a video.
	


	From here:
	https://github.com/jayleicn/moment_detr#run-predictions-on-your-own-videos-and-queries

	git clone git@github.com:jayleicn/moment_detr.git
	cd moment_detr
	pip3 install ffmpeg-python ftfy regex torch

	wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O "/Users/carlos.aguilar/Documents/EF_repos/moment_detr/run_on_video/clip/bpe_simple_vocab_16e6.txt.gz"
	
	python3 run_on_video/run.py


This will load the Moment-DETR model checkpoint trained with CLIP image and text features, 
and make predictions for the video RoripwjYFp8_60.0_210.0.mp4 with its associated query in run_on_video/example/queries.jsonl.



from run_on_video.run import run_example as rexa
rexa()

'''


import torch

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.data_utils import VideoLoader
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np


from run_on_video.run import run_example as rexa
from run_on_video.run import MomentDETRPredictor
# fully automated
rexa()



# some control over the process
from utils.basic_utils import load_jsonl
video_path = "/Users/carlos.aguilar/Documents/EF_repos/moment_detr/run_on_video/example/RoripwjYFp8_60.0_210.0.mp4"
query_path = "run_on_video/example/queries.jsonl"
queries = load_jsonl(query_path)
query_text_list = [e["query"] for e in queries]
ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

import carlos_utils.file_utils as fu
fu.printJSON(queries)


# run predictions
print("Build models...")
clip_model_name_or_path = "ViT-B/32"
# clip_model_name_or_path = "tmp/ViT-B-32.pt"
moment_detr_predictor = MomentDETRPredictor(
    ckpt_path=ckpt_path,
    clip_model_name_or_path=clip_model_name_or_path,
    # device="cuda"
    device="cpu"
)
print("Run prediction...")
predictions = moment_detr_predictor.localize_moment(
    video_path=video_path, query_list=query_text_list)

for idx, query_data in enumerate(queries):
  print("-"*30 + f"idx{idx}")
  print(f">> query: {query_data['query']}")
  print(f">> video_path: {video_path}")
  print(f">> GT moments: {query_data['relevant_windows']}")
  print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
        f"{predictions[idx]['pred_relevant_windows']}")
  print(f">> GT saliency scores (only localized 2-sec clips): {query_data['saliency_scores']}")
  print(f">> Predicted saliency scores (for all 2-sec clip): "
        f"{predictions[idx]['pred_saliency_scores']}")



# Try another query
query_text_list_2 = ['person walks in a street market']
predictions_2 = moment_detr_predictor.localize_moment(video_path=video_path, query_list=query_text_list_2)
fu.printJSON(predictions_2)



#######
#######

video_path_2 = '/Users/carlos.aguilar/Documents/EF_Lumiere/EFxAWS/mp4/MOB_16.4.3.1.1_v2.mp4'
# max lengh of videos 150 seconds
import moviepy.editor as mp
clip = mp.VideoFileClip(video_path_2)

sb_clip = clip.subclip(0, 130)

path_to_resized_video = video_path_2.replace('.mp4', 'subclip.mp4')
sb_clip.write_videofile(path_to_resized_video)
sb_clip.write_videofile(path_to_resized_video, codec='libx264')


# run predictions
print("Build models...")
clip_model_name_or_path = "ViT-B/32"
moment_detr_predictor = MomentDETRPredictor(
    ckpt_path=ckpt_path,
    clip_model_name_or_path=clip_model_name_or_path,
    # device="cuda"
    device="cpu"
)
print("Run prediction...")
# query_text_list_3 = ['people playing sports', 'pasta']

query_text_list_3 = ['woman speaking']
predictions = moment_detr_predictor.localize_moment(
    video_path=path_to_resized_video, query_list=query_text_list_3)


for this_query in predictions:
  fu.printJSON(this_query)
