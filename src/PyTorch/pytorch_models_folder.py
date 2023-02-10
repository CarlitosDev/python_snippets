'''
pytorch_models_folder.py
'''

import torch


default_folder = torch.hub.get_dir()
# this can be seen in the environment variable $TORCH_HOME (can be overridden)


