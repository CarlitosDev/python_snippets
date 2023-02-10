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
    UniformTemporalSubsample
)


# Load Pre-trained Model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# Set to eval mode and move to desired device
device = 'cpu'
model = model.to(device)
model = model.eval()
# Set to eval mode and move to desired device
#model = model.eval()




#load labels
path_to_kinetics = '/Users/carlos.aguilar/Documents/ComputerVision/kinetics/kinetics_classnames.json'
with open(path_to_kinetics, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")



# Load Video
this_video = '/Users/carlos.aguilar/Documents/EF_Content/EFxAWS/mp4/MOB_16.6.4.1.1_x.mp4'
video = EncodedVideo.from_path(this_video)
print(f'Current video lasts for {video.duration} seconds')
start_window = float(round(0.25*video.duration))
end_window = start_window + round(0.1*video.duration)
print(f'Analysing from {start_window} to {end_window} seconds')

# From Rekognition (.../EF_Content/EFxAWS/lumiere_data/GeneralEnglish_analysis/MOB_16.6.4.1.1_x_rekognition.json)
VideoMetadata = {
    "Codec": "h264",
    "DurationMillis": 208641,
    "Format": "QuickTime / MOV",
    "FrameRate": 29.970178604125977,
    "FrameHeight": 360,
    "FrameWidth": 640}


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
# If you want to try another of the torch hub models you will need to modify this transform
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# The duration of the input clip is also specific to the model.
#clip_duration = (num_frames * sampling_rate)/frames_per_second


# Get Clip
clip_start_sec = start_window # secs
clip_duration = end_window # secs
video_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration)
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
#inputs = inputs.to(device)

# Pass the input clip through the model
preds = model(inputs[None, ...])

# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
print("Predicted labels: %s" % ", ".join(pred_class_names))