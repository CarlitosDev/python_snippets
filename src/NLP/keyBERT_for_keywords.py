'''
keyBERT_for_keywords.py


source ~/.bash_profile && python3 -m pip install keybert

'''

from keybert import KeyBERT

# from /Users/carlos.aguilar/Documents/EF_Content/EFxAWS/lumiere_data/GeneralEnglish_analysis/0a.6 Scene 2_transcribe.json
doc = """Mm. So tell me about your date with Laura. Well, she's nice. Nice. Yeah, I like it. I like her eyes. She has beautiful blue eyes, long blonde hair as he has a cute smile. Ah, Ken, Jim."""
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
keywords = kw_model.extract_keywords(doc)


#You can set keyphrase_ngram_range to set the length of the resulting keywords/keyphrases:
kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)


# To extract keyphrases
keyphrase_length = 2
kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, keyphrase_length), stop_words=None)

# diversify the results
kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_maxsum=True, nr_candidates=20, top_n=5)