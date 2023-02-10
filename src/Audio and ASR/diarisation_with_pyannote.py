'''

diarisation_with_pyannote.py

brew install libsndfile
pip3 install cffi speechbrain

pip3 install https://github.com/pyannote/pyannote-audio/archive/develop.zip

pip3 install pyannote.audio==1.1.1

Paper:
https://arxiv.org/pdf/1911.01255.pdf

'''



mp3_filepath = '/Users/carlos.aguilar/Documents/EF_EVC_videos/videos/24.01.2022/72a48689-e72b-4439-9888-3e792f35aea9/72a48689-e72b-4439-9888-3e792f35aea9b.mp3'
wav_filepath = mp3_filepath.replace('mp3','wav')

from pydub import AudioSegment
audio_object = AudioSegment.from_mp3(mp3_filepath)
# audio_object.export(wav_filepath, format="wav")


audio_object_downsampled = audio_object.set_frame_rate(16000)
# set to 8 bits
audio_object_downsampled.set_sample_width(1)
#
# audio_object.set_channels(1)
# audio_object_downsampled.export(wav_filepath, format="wav")




# pydub does things in milliseconds
fromMinsToMS = 60*1e3
slot_start = 30*fromMinsToMS
slot_end = 31*fromMinsToMS
audio_object_segment = audio_object[slot_start:slot_end]
audio_object_downsampled = audio_object_segment.set_frame_rate(16000)
# set to 8 bits
audio_object_downsampled.set_sample_width(1)
#
# audio_object.set_channels(1)
audio_object_downsampled.export(wav_filepath, format="wav")



# We are about to run a full speaker diarization pipeline, that includes 
# - speech activity detection, 
# - speaker change detection, 
# - speaker embedding,
# - and a final clustering step.

usePyannote = False
# use pyannote directly or Pytorch
# I am getting some issues with using the library directly. It seems that the methods 
# have changed places in different versions.
if usePyannote:
    from pyannote.audio import Pipeline
    # from pyannote.pipeline import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
else:
    import torch
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
    


# one can use their own file like this...
test_file = {'uri': 'filename', 'audio': wav_filepath}
diarization = pipeline(test_file)




#diarization.chart()


import carlos_utils.file_utils as fu
fu.printJSON(diarization.for_json())



# let's visualize the diarization output using pyannote.core visualization API
from matplotlib import pyplot as plt
from pyannote.core import Segment, notebook



# only plot one minute (between t=0s and t=40s)
notebook.crop = Segment(0, 40)

fig_h = 10
fig_w = 18
fig, axes = plt.subplots(1,1, figsize=(fig_w, fig_h))
this_ax = axes






# 2nd row: pipeline output
notebook.plot_annotation(diarization, ax=this_ax, time=False)
this_ax.text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)


# for turn, track, speaker in diarization.itertracks(yield_label=True):
#     turn, track, speaker



cohort_title = f'Diarisation using Pyannote'
y_axis_label = 'Average minutes self study (lessons)'
axis_1_title = f'Self study for {cohort_title}'

this_ax.grid(True)
this_ax.margins(0,-0.25)
# this_ax.legend()

# this_ax.set_title(axis_1_title)
# this_ax.set_xlabel(x_axis_label)
# this_ax.set_ylabel(y_axis_label)

plt.tight_layout()
plt.show(block=False)