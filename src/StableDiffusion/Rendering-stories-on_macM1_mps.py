# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
'''We recommend you use attention slicing to reduce memory pressure during inference and 
prevent swapping, particularly if your computer has lass than 64 GB of system RAM, 
or if you generate images at non-standard resolutions larger than 512 × 512 pixels. 

Attention slicing performs the costly attention operation in multiple steps instead of all at once. 
It usually has a performance impact of ~20% in computers without universal memory, 
but we have observed better performance in most Apple Silicon computers, **unless you have 64 GB or more**.
'''
# pipe.enable_attention_slicing()

prompt = "Hyperrealistic. Boy 4 year old riding a bicycle in the forest"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
import carlos_utils.file_utils as fu
story_file = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/The Thirsty Ghost/chatGPT.txt'
story_prompt = fu.readTextFile(story_file)
story_paragraphs = [paragraph for paragraph in story_prompt.split('\n') if paragraph != '']

this_prompt = story_paragraphs[0]
import pyperclip as pp
pp.copy(this_prompt)
image = pipe(prompt).images[0]

import carlos_utils.image_utils as imgu
imgu.show_image(image)