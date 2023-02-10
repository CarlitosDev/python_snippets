'''
  facebook's
  bart_large_text_classification.py
  from https://huggingface.co/facebook/bart-large-mnli


  This model helps to classify sentences into topics.

'''


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classification = classifier(sequence_to_classify, candidate_labels)



candidate_labels =['issue', 'problem','excellent', 'technical issues']


#df['labels'] = df.apply(lambda x: classifier(x.message, candidate_labels, multi_class=True), axis=1)

