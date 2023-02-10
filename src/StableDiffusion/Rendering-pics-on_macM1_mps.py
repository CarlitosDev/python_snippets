# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")



# Open Journey (midJourney style)
from diffusers import StableDiffusionPipeline
import torch
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
pipe = pipe.to("mps")
prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"

prompt = '''full body cyborg| full-length portrait| detailed face| symmetric| steampunk| cyberpunk| cyborg| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''

image = pipe(prompt).images[0]
import carlos_utils.image_utils as imgu
imgu.show_image(image)

image.save("./retro_cars.png")




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


import os
import carlos_utils.image_utils as imgu
from diffusers import DiffusionPipeline
# pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float16, revision="fp16")
model_id = "runwayml/stable-diffusion-v1-5"
model_id = "prompthero/openjourney"
pipe = DiffusionPipeline.from_pretrained(model_id, custom_pipeline="stable_diffusion_mega").to("mps")
pipe.enable_attention_slicing()

output_folder = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/openjourney tests/fantasy'
prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
img_path = os.path.join(output_folder, 'fantasy forest landscape.jpg')
images.save(img_path)


### Text-to-Image
prompt = '''full body cyborg| full-length portrait| detailed face| symmetric| steampunk| cyberpunk| cyborg| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)

prompt = '''fox | hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)


output_folder = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/openjourney tests/fox'
prompt = '''cybernetic fox | mdjrny-v4 style'''
img_path = os.path.join(output_folder, 'cybernetic fox.jpg')
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
images.save(img_path)


prompt = '''cybernetic fox | hyperrealistic| cinematic lighting | mdjrny-v4 style'''
img_path = os.path.join(output_folder, 'cybernetic fox-cinema.jpg')
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
images.save(img_path)


prompt = '''cybernetic fox | hyperrealistic| mdjrny-v4 style'''
img_path = os.path.join(output_folder, 'cybernetic fox-hyperrealistic.jpg')
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
images.save(img_path)


prompt = '''cybernetic fox | manga| mdjrny-v4 style'''
img_path = os.path.join(output_folder, 'cybernetic fox-manga.jpg')
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
images.save(img_path)

prompt = '''cybernetic fox | manga'''
img_path = os.path.join(output_folder, 'cybernetic fox-only manga.jpg')
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
images.save(img_path)



import carlos_utils.file_utils as fu
output_folder = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/openjourney tests/boys'
prompt = '''5 year old boy with blue eyes and dark hair | A 2 year old with fair curly hair | brave explorers | haunted house | sharp focus | illustration | highly detailed | manga style'''
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
img_path = os.path.join(output_folder, 'boys-manga.jpg')
images.save(img_path)


prompt = '''5 year old boy with blue eyes and dark hair, 2 year old with light curly hair, brave explorers, haunted house, sharp focus, illustration, highly detailed'''
images = pipe.text2img(prompt).images[0]
imgu.show_image(images)
img_path = os.path.join(output_folder, 'boys-manga2.jpg')
images.save(img_path)

prompt = '''haunted house, sharp focus, illustration, highly detailed'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'boys-manga3.jpg')
images.save(img_path)

prompt = '''45 year old | female super doctor| blonde |full-length portrait| detailed face| symmetric| steampunk| cyberpunk| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'doctor3.jpg')
images.save(img_path)

prompt = '''45 year old | female super doctor| blonde |full-length portrait| detailed face| symmetric| cyberpunk| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'doctor4.jpg')
images.save(img_path)


prompt = '''female | super doctor| blonde |full-length portrait| detailed face| symmetric| steampunk| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'doctor5.jpg')
images.save(img_path)

prompt = '''female | super doctor| blonde |full-length portrait| detailed face| symmetric| intricate detailed| to scale| hyperrealistic| cinematic lighting| digital art| concept art| mdjrny-v4 style'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'doctor6.jpg')
images.save(img_path)

prompt ='''mdjrny-v4 style use golden ratio to imagine most beautiful banana, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, 8k'''
images = pipe.text2img(prompt).images[0]
# imgu.show_image(images)
img_path = os.path.join(output_folder, 'banana_1.jpg')
images.save(img_path)

### Image-to-Image
path_to_img = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/IMG_0602 copy.jpg'
path_to_img ='/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/sample images/sketch-mountains-input.jpg'
init_image = imgu.load_image_as_PIL(path_to_img)

prompt = "A fantasy landscape, trending on artstation"
images = pipe.img2img(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
imgu.show_image(images[0])


### Inpainting
import os
base_folder = '/Users/carlos.aguilar/Library/CloudStorage/GoogleDrive-carlos.aguilar.palacios@gmail.com/My Drive/StoryBook/sample images'
img_url = os.path.join(base_folder, "overture-creations-5sI6fQgYIuo.png")
mask_url = os.path.join(base_folder, "overture-creations-5sI6fQgYIuo_mask.png")
init_image = imgu.load_image_as_PIL(img_url).resize((512, 512))
mask_image = imgu.load_image_as_PIL(mask_url).resize((512, 512))

prompt = "a fox sitting on a bench"
images = pipe.inpaint(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.75).images
imgu.show_image(images[0])







# This one crashes
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel
import torch

feature_extractor = CLIPFeatureExtractor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)

# guided_pipeline = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     custom_pipeline="clip_guided_stable_diffusion",
#     clip_model=clip_model,
#     feature_extractor=feature_extractor,
#     torch_dtype=torch.float16,
# ).to("mps")

guided_pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor
).to("mps")

guided_pipeline.enable_attention_slicing()

prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"

image = guided_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_guidance_scale=100,
        num_cutouts=4,
        use_cutouts=False
    )


images = pipe.text2img(prompt).images[0]
imgu.show_image(images)


generator = torch.Generator().manual_seed(0)
images = []
for i in range(4):
    image = guided_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_guidance_scale=100,
        num_cutouts=4,
        use_cutouts=False,
        generator=generator,
    ).images[0]
    image.save(f"{base_folder}/clip_guided_sd/image_{i}.png")
    images.append(image)
    
# save images locally
for i, img in enumerate(images):
    img.save(f"{base_folder}/clip_guided_sd/image_{i}.png")