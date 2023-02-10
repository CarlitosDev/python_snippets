'''

course_exercises.py


https://huggingface.co/course/chapter1?fw=pt

Hugging Face ecosystem:
    🤗 Transformers
    🤗 Datasets
    🤗 Tokenizers
    🤗 Accelerate 
+ the Hugging Face Hub. 


'''

'''
main NLP tasks

> Classifying whole sentences: Getting the sentiment of a review, detecting if an email is spam, 
    determining if a sentence is grammatically correct or whether two sentences are logically related or not.

> Classifying each word in a sentence: Identifying the grammatical components 
    of a sentence (noun, verb, adjective), or the named entities (person, location, organization).

> Generating text content: Completing a prompt with auto-generated text, 
    filling in the blanks in a text with masked words

> Extracting an answer from a text: Given a question and a context, 
    extracting the answer to the question based on the information provided in the context.

> Generating a new sentence from an input text: Translating a text 
    into another language, summarizing a text.


For us:
Topic modelling.

'''


'''
    Chapter 1
    https://huggingface.co/course/chapter1?fw=pt
'''

import os
import pandas as pd
import re 
import utils.file_utils as fu
import utils.utils_root as ur
import utils.content_utils as cnu

# Let's load up some real data.
activityId = 278539
activityId = 145591
filename = f'Activity_{activityId}.pickle'
output_path = fu.fullfile(ur.local_data_repo_root(), 'MoviePresentation_V2')
filePath = fu.fullfile(output_path, filename)
movie_presentation_info = fu.readPickleFile(filePath)






'''
    It seems that pretty much anything can be done from the pipeline
'''


from transformers import pipeline


####
#   sentiment analysis
####

# this data is coming from the ACR of an actual lesson. See EnglishLive/learningScore/example_use_profile_data.py
comment_from_lesson = '''It was a pleasure talking to you again today Daniela! I hope you found the lesson on 'Stages of learning' an enjoyable and helpful experience and I look forward to meeting again in class soon. Good luck with your English studies! If you have any questions, please feel free to contact me at dffe.sdsdsd@ef.com'''
classifier_sa = pipeline("sentiment-analysis")

classifier_sa(comment_from_lesson)


####
#   text classification
####

# zero shot to classify text into categories
classifier_zs = pipeline("zero-shot-classification")
classifier_zs(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)



####
#   text-generation
####


text_prompt = "In this course, we will teach you how to"
generator = pipeline("text-generation")
generated_text = generator(text_prompt)[0]['generated_text']


# text generation with a particular model *distilgpt2*
generator = pipeline("text-generation", model="distilgpt2")
hf_response = generator(
    text_prompt,
    max_length=30,
    num_return_sequences=2,
)
generated_text_distilgpt2 = hf_response[0]['generated_text']

# text generation with the open source alternative to GPT-3, GPT neo. Picking the smallest model available.
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
hf_response_gptneo = generator(text_prompt, 
    do_sample=True, 
    min_length=50
)
generated_text_gptNeo = hf_response_gptneo[0]['generated_text']




####
#   question-answering
####

'''
this context comes from parsing the activityId 273629 with the code in 
section K from file EFCatalyst/generate_draft_schemas.py
'''

nlp = pipeline('question-answering')

this_context = "Hey! Come and meet my friends.OK.OK.Michelle, this is Frank and Mary. Frank and Mary, this is Michelle.Hi. Nice to meet you.Nice to meet you.Hi, Michelle.Hi.Michelle's from Brighton.Mary's from London, Frank's from Los Angeles, but they live in London.Frank and Mary, this is James. James is from Birmingham, the UK.Nice to meet you.Nice to meet you.Hi.Hi.Paul is from London.Hi. How's it going?Hi.Hi.Nice to meet you.You, too."
this_question = "Where is Michelle from?"
result = nlp(context=this_context, question=this_question)

this_question = 'Where is Frank from?'
result_q2 = nlp(context=this_context, question=this_question)


# use any model from the Hub in a pipeline
# Go to the Model Hub https://huggingface.co/models and point at them
# bert-large-uncased-whole-word-masking-finetuned-squad


####
#   fill-mask
####

# The idea of this task is to fill in the blanks in a given text

unmasker = pipeline("fill-mask")
unmasker_response = unmasker("This course will teach you all about <mask> models.", top_k=2)


####
#   Named entity recognition
####

ner = pipeline("ner", grouped_entities=True)
ner_response = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


####
#   Text summarisation
####
summarizer = pipeline("summarization")
summarizer("""
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.""")

####
#   Text translation
####
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")



#####################################################################################




