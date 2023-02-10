transcription = (
    "Playing games has always been thought to be important to "
    "the development of well-balanced and creative children; "
    "however, what part, if any, they should play in the lives "
    "of adults has never been researched that deeply. I believe "
    "that playing games is every bit as important for adults "
    "as for children. Not only is taking time out to play games "
    "with our children and other adults valuable to building "
    "interpersonal relationships but is also a wonderful way "
    "to release built up tension."
)





textstat.automated_readability_index(transcription)
textstat.dale_chall_readability_score(transcription)

textstat.linsear_write_formula(transcription)

textstat.text_standard(transcription)
textstat.fernandez_huerta(transcription)
textstat.szigriszt_pazos(transcription)
textstat.gutierrez_polini(transcription)
textstat.crawford(transcription)

#Spanish-specific tests
>>> textstat.fernandez_huerta(transcription)
>>> textstat.szigriszt_pazos(transcription)
>>> textstat.gutierrez_polini(transcription)
>>> textstat.crawford(transcription)

# Syllable Count
textstat.syllable_count(transcription)

# Lexicon Count
textstat.lexicon_count(transcription, removepunct=True)

# Sentence Count
textstat.sentence_count(transcription)

# The following table can be helpful to assess the ease of readability in a document.
# While the maximum score is 121.22, there is no limit on how low the score can be. A negative score is valid.
# https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
textstat.flesch_reading_ease(transcription)


# Flesch-Kincaid Grade of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
textstat.flesch_kincaid_grade(transcription)

# FOG index of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# https://en.wikipedia.org/wiki/Gunning_fog_index
textstat.gunning_fog(transcription)

# MOG index of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# Texts of fewer than 30 sentences are statistically invalid, because the SMOG formula was normed on 30-sentence samples. textstat requires at least 3 sentences for a result.
textstat.smog_index(transcription)

# ARI (Automated Readability Index) which outputs a number that approximates the grade level needed to comprehend the text.
# For example if the ARI is 6.5, then the grade level to comprehend the text is 6th to 7th grade.
# https://en.wikipedia.org/wiki/Automated_readability_index
textstat.automated_readability_index(transcription)

# Coleman-Liau Formula. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
textstat.coleman_liau_index(transcription)

# Dale-Chall Readability Score
# it uses a lookup table of the most commonly used 3000 English words. Thus it returns the grade level using the New Dale-Chall Formula
# https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
textstat.dale_chall_readability_score(transcription)

# Readability Consensus based upon all the above tests
textstat.text_standard(transcription, float_output=False)

# 
textstat.crawford(transcription)

textstat.difficult_words(transcription)