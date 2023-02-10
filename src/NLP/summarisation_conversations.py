'''

  I have tried this on the 02.03.2022. 
  
  I have to give it a go with the classroom conversations.
  It looks quite interesting.
  

  ~/.cache/huggingface/transformers

'''

from transformers import pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
'''

summarisation = summarizer(conversation)
print(summarisation[0]['summary_text'])
# only is the conversation is > max_lenght
# summarisation = summarizer(conversation, max_length=150)




# from transformers import BartTokenizer, BartForConditionalGeneration
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")



import carlos_utils.file_utils as fu
path_to_picke = '/Users/carlos.aguilar/Documents/EF_EVC_videos_hyperclass/videolessons/09.02.2022/8f210358-5c68-463a-83a8-6e1d63688bea/lesson_analysis/analysis_8f210358-5c68-463a-83a8-6e1d63688bea.pickle'
transcription_analysis = fu.readPickleFile(path_to_picke)
df_transcribe_ext = transcription_analysis.get('df_transcribe_ext')
df_speaker_conversational_chunk = transcription_analysis['df_speaker_conversational_chunk']
df_speaker_conversational_chunk.tagged_speaker = df_speaker_conversational_chunk.tagged_speaker.astype(str)
df_speaker_conversational_chunk['y_position'] = 1
lesson_summary = transcription_analysis['lesson_summary']