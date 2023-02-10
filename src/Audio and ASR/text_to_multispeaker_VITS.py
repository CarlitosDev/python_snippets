'''

text_to_multispeaker_VITS.py

Synthesise text

From: https://github.com/jaywalnut310/vit



SETUP:
source ~/.bash_profile && python3

cd monotonic_align
python3 setup.py build_ext --inplace
cd ..
brew install espeak
(Linux: sudo apt-get install espeak -y)

# wget -O pretrained_ljs.pth "https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT"
# That doesn't work because of the approval. Just paste in the browser 
# https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT
# and download.

# Same for pretrained_vctk.pth
# https://drive.google.com/uc?id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru
pip3 install sounddevice



'''
import os
import torch
import commons
from text import text_to_sequence
from text.symbols import symbols
import utils
from models import SynthesizerTrn
from scipy.io.wavfile import write

from understand_the_VCTK_dataset import get_the_damn_codes
train_index_to_id, train_id_to_index, train_speaker_to_age, train_speaker_to_gender, train_speaker_to_accent = get_the_damn_codes()

from scipy.io.wavfile import write as wav_write
def save_audio_piece(audio_matrix: 'numpy.ndarray', filename, sampling_rate):
  wav_write(filename, sampling_rate, audio_matrix)



text_to_synthesise = "Hi, my name is Carlos and I have just joined E F English Live. I would like to hear myself speaking after taking the course."


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm




## >>> 2
# Multi speaker model to synthetise text
hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")

'''

http://www.udialogue.org/download/cstr-vctk-corpus.html


This CSTR VCTK Corpus includes speech data uttered by 109 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive. The newspaper texts were taken from Herald Glasgow, with permission from Herald & Times Group. Each speaker has a different set of the newspaper texts selected based a greedy algorithm that increases the contextual and phonetic coverage. The rainbow passage and elicitation paragraph are the same for all speakers. The rainbow passage can be found at International Dialects of English Archive: (http://web.ku.edu/~idea/readings/rainbow.htm). The elicitation paragraph is identical to the one used for the speech accent archive (http://accent.gmu.edu). The details of the the speech accent archive can be found at http://www.ualberta.ca/~aacl2009/PDFs/WeinbergerKunath2009AACL.pdf All speech data was recorded using an identical recording setup: an omni-directional microphone (DPA 4035), 96kHz sampling frequency at 24 bits and in a hemi-anechoic chamber of the University of Edinburgh. All recordings were converted into 16 bits, were down-sampled to 48 kHz based on STPK, and were manually end-pointed. This corpus is aimed for HMM-based text-to-speech synthesis systems, especially for speaker-adaptive HMM-based speech synthesis that uses average voice models trained on multiple speakers and speaker adaptation technologies.


For English speech synthesis, the  VCTK  corpus  consists  of  44 hours  of 
speech  data  uttered  by  109  native  speakers  of  En-glish with various accents.

 To ensure the quality of speech data, all voicesin two corpora are recorded 
 in semi-anechoic chambers.  How-ever, limited by the recording cost, 
 the scale of both them can-not  meet  the  needs  of  researchers


This is kind of what I'm looking for
https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/vctk.py


'''

# initialise
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g_ms.eval()

_ = utils.load_checkpoint("pretrained_vctk.pth", net_g_ms, None)

stn_tst = get_text(text_to_synthesise, hps_ms)


# speaker identity
import carlos_utils.file_utils as fu
fu.printJSON(train_speaker_to_accent)


# default 4
speaker_identity = 311
_gender = train_speaker_to_gender[speaker_identity]
_accent = train_speaker_to_accent[speaker_identity]

sid = torch.LongTensor([train_id_to_index[speaker_identity]]) 

with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()


examples_folder = '/Users/carlos.aguilar/Documents/EF_repos/vits (speech synthesis)/carlos_examples'
mspeaker_filename = os.path.join(examples_folder, f'text_2_multispeaker_{speaker_identity}_{_gender}_{_accent}' + '.wav')

save_audio_piece(audio, mspeaker_filename,hps_ms.data.sampling_rate)




## Go for several speakers
examples_folder = '/Users/carlos.aguilar/Documents/EF_repos/vits (speech synthesis)/carlos_examples'
with torch.no_grad():
  for int_speaker_idx in range(1, 40):
    speaker_identity = train_index_to_id[int_speaker_idx]
    _gender = train_speaker_to_gender[speaker_identity]
    _accent = train_speaker_to_accent[speaker_identity]

    sid = torch.LongTensor([int_speaker_idx])
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()

    mspeaker_filename = os.path.join(examples_folder, f'text_2_multispeaker_{speaker_identity}_{_gender}_{_accent}' + '.wav')

    save_audio_piece(audio, mspeaker_filename,hps_ms.data.sampling_rate)
