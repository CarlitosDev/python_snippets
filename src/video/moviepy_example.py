
'''
pip3 install moviepy
pip3 install SpeechRecognition moviepy --ignore-installed
'''


path_to_video = '/Users/carlos.aguilar/Documents/EF_DataScience/video_project/TravelVocabularyVideo.mp4'



import moviepy.editor as mp
clip = mp.VideoFileClip(path_to_video)

path_to_audio = path_to_video.replace('mp4', 'wav')
clip.audio.write_audiofile(path_to_audio)

path_to_mp3_audio = path_to_video.replace('mp4', 'mp3')
clip.audio.write_audiofile(path_to_mp3_audio)

# resize the video

path_to_resized_video = path_to_video.replace('.mp4', '_resized.mp4')
sb_clip = clip.subclip(0, 59)
sb_clip = sb_clip.resize(0.5)
sb_clip.write_videofile(path_to_resized_video, codec='libx264')


import speech_recognition as sr 
r = sr.Recognizer()
audio = sr.AudioFile(path_to_audio)
with audio as source:
  audio_file = r.record(source)
result = r.recognize_google(audio_file)


path_to_audio_contents = path_to_video.replace('mp4', 'txt')
# exporting the result 
with open(path_to_audio_contents, mode ='w') as file: 
   file.write("Recognized Speech:") 
   file.write("\n") 
   file.write(result)




## Another video
path_to_video = '/Users/carlos.aguilar/Documents/EF_DataScience/video_project/10.mp4'
clip = mp.VideoFileClip(path_to_video)

path_to_mp3_audio = path_to_video.replace('mp4', 'mp3')
clip.audio.write_audiofile(path_to_mp3_audio)
