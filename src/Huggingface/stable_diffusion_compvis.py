'''
    stable_diffusion_compvis.py
'''


import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.device("mps")

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5).images[0]  
    
image.save("astronaut_rides_horse.png")



'''
>> this works (01.11.2022)
To run it on MAC
https://invoke-ai.github.io/InvokeAI/installation/INSTALL_MAC/
Also...
https://github.com/einanao/stable-diffusion/tree/apple-silicon

'''

pipe_mps = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, revision="fp16",
    use_auth_token=True,
).to("mps")


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe_mps(prompt, guidance_scale=7.5).images[0]
image.save("mps_astronaut_rides_horse.png")
