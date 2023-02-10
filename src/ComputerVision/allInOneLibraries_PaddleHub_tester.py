'''

PaddleHub
https://github.com/PaddlePaddle/PaddleHub


Tutorials in Chinese
https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/docs/docs_en/visualization.md



Some of the major pre-trained models PaddleHub contains are:
a. Text recognition
b. Image editing & Image generation
c. Face detection, Object detection & Keypoint detection
d. Image segmentation & Image classification


pip3 install paddlepaddle paddlehub --upgrade

For PaddleOCR use my version:
pip3 install git+https://github.com/CarlitosDev/PaddleOCR.git@main

'''


import paddlehub as hub



# A - Face Detection


module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")

this_image = '/Users/carlos.aguilar/Documents/Pictures_Carlos/IMG_6143.jpg'
output_dir = '/Users/carlos.aguilar/Documents/Pictures_Carlos/face_detection'

res = module.face_detection(
paths = [this_image],
visualization=True,
output_dir=output_dir)

import carlos_utils.file_utils as fu
fu.printJSON(res[0])


# B - openpose body estimation

output_dir_pose = '/Users/carlos.aguilar/Documents/Pictures_Carlos/pose_estimation'
fu.makeFolder(output_dir_pose)

module = hub.Module(name="openpose_body_estimation")

res = module.predict(
    img=this_image,
    visualization=True,
    save_path=output_dir_pose)

fu.printJSON(res)


###
#
####
# C - Style transfer
model = hub.Module(name='msgnet', load_checkpoint=None)


this_image = '/Users/carlos.aguilar/Documents/Pictures_Carlos/Luca/IMG_2082.jpg'
output_dir = '/Users/carlos.aguilar/Documents/Pictures_Carlos/style_transfer'

result = model.predict(origin=[this_image], 
style='/Users/carlos.aguilar/Downloads/temp_Vincent.jpeg', 
visualization=True, 
save_path = output_dir
)

fu.printJSON(result)


# pip3 uninstall paddleocr

## D - OCR
## pip3 install paddleocr
# pip3 install git+https://github.com/CarlitosDev/PaddleOCR.git@main



# 1- download the English ultra-lightweight PP-OCRv3 model INFERENCE models from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/ppocr_introduction_en.md#pp-ocrv3
# /Volumes/TheStorageSaver/29.12.2021-EducationFirst/deep_learning_models/paddlehub/ocr/ch_ppocr_mobile_v2.0_cls_infer
# I have manually copied tools to ppocr/tools
from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = '''/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/13.07.2022/d0ed81d2-74bc-4591-abf5-707eb8b7c7c9/scene_detection/Scene  6 - 174.7 seconds. From 00.31.41.500 to 00.34.36.200 - 1747 frames (19015,20762).png'''
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)


from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
font_path = '/Users/carlos.aguilar/Documents/temp_repos/PaddleOCR/doc/fonts/simfang.ttf'
im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
# im_show = Image.fromarray(im_show)
fig, ax = plt.subplots()
im = ax.imshow(im_show)
ax.axis('off')
plt.show()




result_classification = ocr.ocr(img_path, det=False, cls=True)
for line in result_classification:
    print(line)


# only detection
text_detected = ocr.ocr(img_path,rec=False)
for line in text_detected:
    print(line)




