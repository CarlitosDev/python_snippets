'''

face_detection_deepface.py


Update for Deepface a lightweight face recognition and facial attribute analysis framework for python. 

It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib.


GitHub 
https://github.com/serengil/deepface

pip3 install deepface --upgrade

https://m.youtube.com/watch?v=WnUVYQP4h44&list=PLsS_1RYmYQQFdWqxQggXHynP1rqaYXv_E&index=1


https://github.com/serengil/deepface

'''


from deepface import DeepFace
import utils.image_utils as imu
import os
import utils.file_utils as fu



baseFolder = os.path.expanduser(os.path.join('~', 'Google Drive', 'pictures'))



'''
1-
Face Verification - Demo
This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or based64 encoded images is also welcome.
'''


img_1_path = os.path.join(baseFolder, 'IMG_2737.HEIC')
img_2_path = os.path.join(baseFolder, 'IMG_6476.HEIC')

img_1_data = imu.load_heic_image(img_1_path)
img_2_data = imu.load_heic_image(img_2_path)

img_1_jpg_path = img_1_path.replace('HEIC', 'jpg')
img_2_jpg_path = img_2_path.replace('HEIC', 'jpg')

img_1_data.save(img_1_jpg_path, format="JPEG")
img_2_data.save(img_2_jpg_path, format="JPEG")


result = DeepFace.verify(img_1_jpg_path, img_2_jpg_path)

fu.printJSON(result)


# Use other models
'''
FaceNet, VGG-Face, ArcFace and Dlib overperforms than OpenFace, 
DeepFace and DeepID based on experiments. 

On the LFW data set:
    FaceNet /w 512d got 99.65%; 
    FaceNet /w 128d got 99.2%; 
    ArcFace got 99.41%; 
    Dlib got 99.38%; VGG-Face got 98.78%; 
    DeepID got 97.05; 
    OpenFace got 93.80% accuracy scores 
    Human beings could have just 97.53%.
'''

# /Users/carlos.aguilar/.deepface/weights/{facenet_weights.h5}
available_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

result_facenet = DeepFace.verify(img_1_jpg_path, img_2_jpg_path, model_name = available_models[1])
fu.printJSON(result_facenet)


result_DeepFace = DeepFace.verify(img_1_jpg_path, img_2_jpg_path, model_name = 'DeepFace')
fu.printJSON(result_DeepFace)


result_Facenet512 = DeepFace.verify(img_1_jpg_path, img_2_jpg_path, model_name = 'Facenet512')
fu.printJSON(result_Facenet512)

# 1b 
# Face recognition requires applying face verification many times. 
# Herein, deepface has an out-of-the-box find function to handle this action. 
# It's going to look for the identity of input image in the database path and 
# it will return pandas data frame as output.

# this creates a DB in a pickle file
df = DeepFace.find(img_path = img_1_jpg_path, db_path = baseFolder)




# Deepface also comes with a strong facial attribute analysis module including 
# age, gender, facial expression (including angry, fear, neutral, sad, disgust, happy and surprise) 
# and race (including asian, white, middle eastern, indian, latino and black) predictions.

face_analysis = DeepFace.analyze(img_path=img_1_jpg_path, actions = ['age', 'gender', 'race', \
    'emotion', 'facial expression'])

fu.printJSON(face_analysis)
imu.show_imagefile(img_1_jpg_path)


# Streaming and Real Time Analysis - Demo
# You can run deepface for real time videos as well. 
# Stream function will access your webcam and apply both face recognition and facial attribute analysis.
# The function starts to analyze a frame if it can focus a face sequantially 5 frames. 
# Then, it shows results 5 seconds.



# Create embeddings
img_1_embedding = DeepFace.represent(img_1_jpg_path, model_name = 'Facenet')
len(img_1_embedding)



# Face Detectors - Demo

# Face detection and alignment are early stages 
# of a modern face recognition pipeline. 
# Experiments show that just alignment increases the face recognition accuracy
#  almost 1%. OpenCV, SSD, Dlib, MTCNN and RetinaFace detectors are wrapped in deepface. 
# OpenCV is the default detector.

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

#face detection and alignment
detected_face = DeepFace.detectFace(img_1_jpg_path, detector_backend = backends[4])
imu.show_image(detected_face)

