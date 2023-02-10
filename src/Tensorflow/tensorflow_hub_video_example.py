'''
       I don't know how to manipulate the output. I have to read more about this type of models.
'''


import numpy as np
# TF1 version
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# or this one
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub



model_handle = 'https://tfhub.dev/google/tiny_video_net/tvn3/1'
module = hub.Module(model_handle)
print(module.get_signature_names())
print(module.get_input_info_dict())


batch_size = 64
num_frames = 1
image_size = 160
vid_placeholder = tf.placeholder(tf.float32,
                                 shape=(batch_size * num_frames,
                                        image_size, image_size, 3))

# This generates a random video. It should be replaced by a user's video.
# video = load_video(video_path) which should return a video of the above shape.
video = np.random.rand(*vid_placeholder.shape)


predictions = module(video)

