
'''
deepspeech2_paddlepaddle.py

pip3 install paddlepaddle paddleaudio paddlespeech paddlehub paddlenlp --upgrade

# pip3 install paddlespeech==0.1.0rc0


'''


import numpy as np
import paddlehub as hub
from paddlenlp import Taskflow
from paddlespeech.cli import ASRExecutor
import soundfile as sf

import paddlespeech as pds

# asr_model = hub.Module(name='u2_conformer_aishell')

asr_executor = pds.cli.ASRExecutor()
text_correct_model = Taskflow("text_correction")
punc_model = hub.Module(name='auto_punc')




text = asr_executor(file)
text_correction = text_correct_model(text)[0]
cor_text, errors = text_correction['target'], text_correction['errors']
print(f'[Text Correction] errors: {errors}')
punc_text = punc_model.add_puncs(cor_text, device='cpu')[0]

ret = ''
ret += f'[ASR] {text}\n'
ret += f'[COR] {cor_text}\n'
ret += f'[PUN] {punc_text}'


import paddlespeech as pds
from paddlespeech.cli.asr import ASRExecutor