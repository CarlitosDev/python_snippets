'''
	NER_with_Spacy.py

	Download a roBERTa model that has been fine trained for NER (https://spacy.io/models/en#en_core_web_trf)
	source ~/.bash_profile && python3 -m spacy download en_core_web_trf

'''



import spacy
nlp = spacy.load('en_core_web_trf')


transcription = '''Fastly released its Q1-21 performance on Thursday, after which the stock price 
dropped a whopping 27%. The company generated revenues of $84.9 million (35% YoY) vs. $85.1 million market consensus.
Net loss per share was $0.12 vs. an expected $0.11.
These are not big misses but make the company one of the few high-growth cloud players that underperformed 
market expectations. However, the company also lowered its guidance for Q2: Fastly forecasts revenues of $84 - $87 million and a
 net loss of $0.16 - $0.19 per share, compared to the market consensus of $92 million in revenue and 
 a net loss of $0.08 per share, thereby disappointing investors.
Lastly, Adriel Lares will step down as CFO of the company after 5 years.'''

current_doc = nlp(transcription)
# Get the entities
detected_entities = [(entity.text, entity.label_, str(spacy.explain(entity.label_))) for entity in current_doc.ents]

for this_entity in detected_entities:
	print(this_entity)



transcription = '''Adam Elliott was appointed President and CEO of iA Clarington in April 2021. 
He joined iA Clarington as National Sales Manager in 2018 and has over 20 years of experience in 
the fund industry. He previously spent 18 years at Dynamic Funds, holding a variety of senior roles
including Regional Vice-President, Sales for Ontario, and Senior Executive, Business Development.
Adam earned an Honours BA in history from McGill University and holds a number of industry credentials.'''

current_doc = nlp(transcription)


# Get the entities
detected_entities = [(entity.text, entity.label_, str(spacy.explain(entity.label_))) for entity in current_doc.ents]

for this_entity in detected_entities:
	print(this_entity)



#  this is from a lesson
transcription = '''You achieved a 95 score on Environmental concerns (focus on The environment and the future)

Your teacher Melisa S. said
	"Daniela, thank you for choosing a lesson today on "Environmental concerns". I had a good time working with you and enjoyed our conversation. I hope to see you in class in the future. Thanks, Melisa. "

The teacher suggests "Daniela is very confident. We completed tasks 2 and 3. She successfully used the target language within the lesson. She needs to be careful with pronunciation and word choice."
'''

current_doc = nlp(transcription)
# Get the entities
detected_entities = [(entity.text, entity.label_, str(spacy.explain(entity.label_))) for entity in current_doc.ents]

for this_entity in detected_entities:
	print(this_entity)
