'''
ASR_long_files.py
'''



path_to_mp3 = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/evc_API/3ac46cad-ade4-42c0-85ba-9aade117b073b.mp3'


from transformers import pipeline
import carlos_utils.file_utils as fu
import time
queryStart = time.time()


model_id = "facebook/wav2vec2-base-960h"
# model_id = "facebook/wav2vec2-large-xlsr-53"

pipe = pipeline("automatic-speech-recognition", model=model_id)
# stride_length_s is a tuple of the left and right stride length.
# With only 1 number, both sides get the same stride, by default
# the stride_length on one side is 1/6th of the chunk_length_s
output = pipe(path_to_mp3, chunk_length_s=20, stride_length_s=(8, 4))


'''
# from https://github.com/huggingface/transformers/issues/16759
pipe = pipeline("automatic-speech-recognition", 
    model="facebook/wav2vec2-large-960h-lv60-self", 
    tokenizer=processor_with_lm, 
    feature_extractor=processor_with_lm.feature_extractor, 
    decoder=processor_with_lm.decoder, device=0)
'''



  
path_to_pickle = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/ASR_output.pickle'
fu.toPickleFile(output, path_to_pickle)
queryElapsed = time.time() - queryStart
print(f'Processing participants done in {queryElapsed:3.2f} seconds')

# Processing participants done in 671.24 seconds

# Let's inspect the results...
output = fu.readPickleFile(path_to_pickle)['text']





from EVC_utils.asrChunkTranscriber import ChunkTranscriber

path_to_mp3 = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/chunk_transcription/3ac46cad-ade4-42c0-85ba-9aade117b073b.mp3'
asr_model = ChunkTranscriber()

asr_model.process_audiofile(path_to_mp3, window_seconds=10)















from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import Wav2Vec2FeatureExtractor
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")



from transformers import AutoConfig
from transformers.models.wav2vec2 import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained(model_id)
cfg = AutoConfig.from_pretrained(model_id)



from transformers import AutoProcessor, AutoModelForPreTraining
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
processor.tokenizer.get_vocab()
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")



from EVC_utils.asrChunkTranscriber import ChunkTranscriber
# model_name = "facebook/wav2vec2-large-xlsr-53"
model_id = "facebook/wav2vec2-base-960h"
path_to_mp3 = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/evc_API/3ac46cad-ade4-42c0-85ba-9aade117b073b.mp3'
asr_model = ChunkTranscriber(model_name=model_name)



from huggingsound import SpeechRecognitionModel
model_fcb = SpeechRecognitionModel(model_id)


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)


# UPDATE: PyCTCDecode is merged to Transformers!
# pip3 install huggingface_hub transformers --upgrade

import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F


model_id = "patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm"

sample = next(iter(load_dataset("common_voice", "es", split="test", streaming=True)))

import carlos_utils.file_utils as fu
fu.printJSON(sample)

resampled_audio = F.resample(torch.tensor(sample["audio"]["array"]), 48_000, 16_000).numpy()

model = AutoModelForCTC.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

input_values = processor(resampled_audio, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values).logits


transcription = processor.batch_decode(logits.numpy()).text
# => 'bien y qué regalo vas a abrir primero'




# prepare text for kenLM
import carlos_utils.file_utils as fu
path_to_json = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/evc_API/3ac46cad-ade4-42c0-85ba-9aade117b073b.json'
json_data = fu.readJSONFile(path_to_json)
transcription = json_data['results']['transcripts'][0]['transcript']

path_to_txt = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/languageModels/transcription.txt'
fu.writeTextFile(transcription, path_to_txt)

