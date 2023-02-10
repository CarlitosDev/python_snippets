'''

PyTorchVideo is a deeplearning library with a focus on video understanding work. 


source ~/.bash_profile && python3 -m pip install pytorchvideo

source ~/.bash_profile && python3 -m pip install pytorchvideo -U

Follows:
https://github.com/facebookresearch/pytorchvideo/blob/master/tutorials/torchhub_inference_tutorial.ipynb

Downloading: 
"https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth" 
to /Users/carlos.aguilar/.cache/torch/hub/checkpoints/SLOW_8x8_R50.pyth

!wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json




Models:
(A) slow_r50: Resnet Style Video classification networks pretrained on the Kinetics 400 dataset
https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
Christoph Feichtenhofer et al, “SlowFast Networks for Video Recognition” 
https://arxiv.org/pdf/1812.03982.pdf

(B) X3D: X3D networks pretrained on the Kinetics 400 dataset (29.4MB)
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
Christoph Feichtenhofer, “X3D: Expanding Architectures for Efficient Video Recognition.” 
https://arxiv.org/abs/2004.04730


(C) Slowfast: SlowFast networks pretrained on the Kinetics 400 dataset (264MB)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

Christoph Feichtenhofer et al, “SlowFast Networks for Video Recognition” 
https://arxiv.org/pdf/1812.03982.pdf


'''

# Import all the required components
import json
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict


# Load Pre-trained Model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
device = 'cpu'
model = model.to(device)
model = model.eval()




#load labels
path_to_kinetics = '/Users/carlos.aguilar/Documents/ComputerVision/kinetics/kinetics_classnames.json'
with open(path_to_kinetics, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")



# slowfast transform
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second




# Load Video
video_path = '/Users/carlos.aguilar/Documents/ComputerVision/pytorchvideo_resources/archery.mp4'
# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration 

# Initialize an EncodedVideo helper class
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]



# Pass the input clip through the model 
preds = model(inputs)

# Get the predicted classes 
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
print("Predicted labels: %s" % ", ".join(pred_class_names))

# Predicted labels: archery, answering questions, applying cream, abseiling, air drumming
# with the same probability...