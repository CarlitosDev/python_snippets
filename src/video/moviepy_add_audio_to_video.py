
'''
pip3 install SpeechRecognition moviepy --ignore-installed


'''


path_to_video_A = '/Users/carlos.aguilar/Documents/EF_DataScience/deep_fakes/IMG_9965.MOV'
path_to_video_B = '/Users/carlos.aguilar/Documents/EF_DataScience/deep_fakes/mateo_talking.mp4'




import moviepy.editor as mp



# Extract the audio from the first video
clip = mp.VideoFileClip(path_to_video_A)
audioclip = clip.audio
audio_codec = audioclip.reader.acodec

# Add it to the second video
clip_B = mp.VideoFileClip(path_to_video_B)
clip_B.audio = audioclip
path_to_video_C = video_output_name_recomposed.replace('.mp4', '_sound.mp4')
clip_B.write_videofile(path_to_video_C)



## Old solution
# Extract the audio from the first video
clip = mp.VideoFileClip(path_to_video_A)
path_to_mp3_audio = path_to_video_A.replace('MOV', 'mp3')
clip.audio.write_audiofile(path_to_mp3_audio)

audioclip = mp.AudioFileClip(path_to_mp3_audio)


# Add it to the second video
clip_B = mp.VideoFileClip(path_to_video_B)
new_audioclip = mp.CompositeAudioClip([audioclip])
clip_B.audio = new_audioclip

path_to_video_C = path_to_video_B.replace('.mp4', '_sound.mp4')
clip_B.write_videofile(path_to_video_C)