
'''
pip3 install youtube-transcript-api


Get the transcription of this cryptocurrency tutorial

'''
import carlos_utils.file_utils as fu
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
youtube_url = 'rYQgy8QDEBI'




# assigning srt variable with the list
# of dictonaries obtained by the get_transcript() function
lst_transcription = YouTubeTranscriptApi.get_transcript(youtube_url, languages=('es','en'))
  
# prints the result
print(lst_transcription[0])

df_transcription = pd.DataFrame(lst_transcription)


fu.to_random_excel_file(df_transcription)




youtube_url = 'hXkiAfjFtgU'
# assigning srt variable with the list
# of dictonaries obtained by the get_transcript() function
lst_transcription = YouTubeTranscriptApi.get_transcript(youtube_url, languages=('es','en'))
  
# prints the result
print(lst_transcription[0])

import pandas as pd
df_transcription = pd.DataFrame(lst_transcription)
df_transcription['start_min'] = df_transcription.start/60
fu.to_random_excel_file(df_transcription)



## Databricks talk
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import carlos_utils.file_utils as fu
youtube_url = 'KWDVnwmQ0V8'
lst_transcription = YouTubeTranscriptApi.get_transcript(youtube_url, languages=('es','en'))
  
# prints the result
print(lst_transcription[0])
df_transcription = pd.DataFrame(lst_transcription)
df_transcription['start_min'] = df_transcription.start/60
fu.to_random_excel_file(df_transcription)


# all in one paragraph.
current_paragraph = ''
for this_sentence in lst_transcription:
  current_paragraph += (' ' + this_sentence['text'])
  
import pyperclip as pp
pp.copy(current_paragraph)


# The duration includes the silences and more than one sentence.

# timeout_seconds = 1.00

# idx_paragraph = -1
# previous_sentence_end_time = -1000.0
# paragraph_start_time = 0.0 
# all_paragraphs = []
# current_paragraph = ''
# paragraph_end_time = 0.0

# for this_sentence in lst_transcription:

#   if (this_sentence['start']-previous_sentence_end_time) > timeout_seconds:
    
#     idx_paragraph +=1
#     if idx_paragraph > 0:
#         all_paragraphs.append({'text': current_paragraph, \
#           'start': paragraph_start_time, 'end': paragraph_end_time})

#     paragraph_start_time = this_sentence['start']
#     current_paragraph = this_sentence['text']
#     paragraph_end_time = paragraph_start_time + this_sentence['duration']
#     previous_sentence_end_time = this_sentence['start'] + this_sentence['duration']

#   else:
#       current_paragraph += (' ' + this_sentence['text'])
#       paragraph_end_time += this_sentence['duration']
#       previous_sentence_end_time = this_sentence['start'] + this_sentence['duration']
    
    


# this_sentence = lst_transcription[0]
# this_sentence = lst_transcription[1]



#     # start a paragraph
#     current_paragraph += this_sentence['text']
#     paragraph_end_time = paragraph_start_time + this_sentence['duration']




#   previous_sentence_end_time = current_sentence_end_time


#   for this_sentence in lst_transcription:
