'''
keyword_extraction_library_rake-nltk.py

source ~/.bash_profile && python3 -m pip install rake-nltk
'''


from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()


transcription = '''Nice shirt, Todd.Oh, thanks, Joan. It's from Italy.Cool.
Yeah. Italy's great. I go there pretty often. 
My uncle’s from Milan.Wow! I love traveling. But I don't have much money right now.
That’s too bad.That’s okay. I do fun things near home.Like what?Well, like going to the theater. 
Umm, and I do a little oil painting.Nice.And I have a little garden.Gardening? I
n an apartment?Oh, just some flowers and things. On the balcony.
Do you ever grow vegetables?Sometimes. I have some tomatoes this year.
Sweet.They are, actually. You’d be surprised.'''.replace('\n', '').replace('?', '? ')


# Extraction given the text.
a = r.extract_keywords_from_text(transcription)

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()

# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences(<list of sentences>)



# To get keyword phrases ranked highest to lowest with scores.
keywords_ranked  = r.get_ranked_phrases_with_scores()

