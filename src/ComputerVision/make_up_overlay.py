'''
make_up_overlay.py

pip3 install pytorchcv
'''

'''
Some guy in Walmart has done it like this
Custom modification of the Bilateral Segmentation Network is used for Real-time Semantic Segmentation to identify our focus areas of lips and hair. Semantic Segmentation requires both rich spatial information and sizeable receptive field. Spatial information is preserved and high-resolution features are generated through the Spatial Path of BiSeNet. Meanwhile, the Context Path helps in identifying sufficient receptive field. The features obtained are thus fused to identify the target areas to be used for further intelligent color shading. We employ Gaussian blur and apply the color masking technique to give it a natural look.
'''

'''
A repo (3 years old) here:
https://github.com/zllrunning/face-makeup.PyTorch
'''


from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("resnet18", pretrained=True)
x = Variable(torch.randn(1, 3, 224, 224))
y = net(x)

