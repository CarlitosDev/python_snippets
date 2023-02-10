import face_recognition
import os
import cv2
import re
from shutil import  move


def fileparts(thisPath):
    [fPath, fName] = os.path.split(thisPath)
    [file, ext] = os.path.splitext(fName)
    return fPath, file, ext


def process_unknown_image(image_to_label, known_face_encodings, known_face_names):

  unknown_image = face_recognition.load_image_file(image_to_label)
  # shrink
  small_unknown_image = cv2.resize(unknown_image, (0, 0), fx=0.25, fy=0.25)

  # Find all the faces and face encodings in the current frame of video
  face_locations = face_recognition.face_locations(small_unknown_image)
  face_encodings = face_recognition.face_encodings(small_unknown_image, face_locations)

  face_names = []
  for face_encoding in face_encodings:
      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
      name = "Unknown"

      # If a match was found in known_face_encodings, just use the first one.
      if True in matches:
          first_match_index = matches.index(True)
          name = known_face_names[first_match_index]
          face_names.append(name)

  return face_names




# Load a sample picture and learn how to recognize it.
base_folder = '/Users/carlos.aguilar/Documents/face recognition tester'
folder_labelled = 'Mateo'
pics_folder = os.path.join(base_folder, folder_labelled)

known_face_names = []
known_face_encodings = []

for dir_path, dir_names, file_names in os.walk(pics_folder):
  for thisFile in file_names:
    image_Mateo = os.path.join(pics_folder, thisFile)
    print(f'Extracting features from {thisFile}...')
    known_image = face_recognition.load_image_file(image_Mateo)
    known_encoding = face_recognition.face_encodings(known_image)
    #
    if known_encoding != []:
      known_face_encodings.append(known_encoding[0])
      known_face_names.append('Mateo Aguilar')
    else:
      os.remove(image_Mateo)





if __name__ == "__main__":

  do_iPhone = False
  do_Cuba   = True




  if do_iPhone:
    baseFolder ='/Volumes/CarlosPictures/iPhone pics 9.12.18/2018'
    for root, dirs, files in os.walk(baseFolder, topdown=True):  
      baseName = os.path.basename(root)
      if 'JPG' in baseName:
        for thisFile in files:
          print(f'Running {thisFile}...', end='')
          this_image = os.path.join(root, thisFile)
          [fPath, file, ext] = fileparts(this_image)
          if 'Mateo' not in file and '.DS_Store' not in file:
            found_faces = process_unknown_image(this_image, known_face_encodings, known_face_names)
            if found_faces != []:
              faces_as_names = '_'.join(map(str, found_faces))
              print(f'matched to {faces_as_names}!')
              newFilePath = os.path.join(fPath, file + '_' + faces_as_names + ext)
              move(this_image, newFilePath)
            else:
              print('')


  if do_Cuba:
    baseFolder = '/Volumes/CarlosPictures/Photos/2k18_Cuba - fotos'
    for root, dirs, files in os.walk(baseFolder, topdown=True):  
      baseName = os.path.basename(root)
      for thisFile in files:
        if '.jpg' in thisFile.lower(): 
          print(f'Running {thisFile}...', end='')
          this_image = os.path.join(root, thisFile)
          [fPath, file, ext] = fileparts(this_image)
          if 'Mateo' not in file and '.DS_Store' not in file:
            found_faces = process_unknown_image(this_image, known_face_encodings, known_face_names)
            if found_faces != []:
              faces_as_names = '_'.join(map(str, found_faces))
              print(f'matched to {faces_as_names}!')
              newFilePath = os.path.join(fPath, file + '_' + faces_as_names + ext)
              move(this_image, newFilePath)
            else:
              print('')