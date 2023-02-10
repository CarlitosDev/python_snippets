'''

  Follow this tutorial
  https://www.tensorflow.org/hub/tutorials/text_to_video_retrieval_with_s3d_milnce

'''


import os

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

import numpy as np
import cv2
from IPython import display
import math


# Load the model once from TF-Hub.
hub_handle = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
hub_model = hub.load(hub_handle)

# work out how to read the downloaded model


def generate_embeddings(model, input_frames, input_words):
  """Generate embeddings from the model from video frames and input words."""
  # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
  vision_output = model.signatures['video'](tf.constant(tf.cast(input_frames, dtype=tf.float32)))
  text_output = model.signatures['text'](tf.constant(input_words))
  return vision_output['video_embedding'], text_output['text_embedding']


