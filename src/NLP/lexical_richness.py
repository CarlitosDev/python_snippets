'''
	lexical_richness.py

	pip3 install lexicalrichness textblob

  https://github.com/LSYS/lexicalrichness
'''

from lexicalrichness import LexicalRichness


text = """Measure of textual lexical diversity, computed as the mean length of sequential words in
                a text that maintains a minimum threshold TTR score.

                Iterates over words until TTR scores falls below a threshold, then increase factor
                counter by 1 and start over. McCarthy and Jarvis (2010, pg. 385) recommends a factor
                threshold in the range of [0.660, 0.750].
                (McCarthy 2005, McCarthy and Jarvis 2010)"""

# https://github.com/LSYS/lexicalrichness#4-attributes
lex = LexicalRichness(text)

print(lex.words)
print(lex.terms)
print(lex.ttr)
print(lex.rttr)
print(lex.cttr)
print(lex.msttr(segment_window=25))
print(lex.mattr(window_size=25))
print(lex.mtld(threshold=0.72))
print(lex.hdd(draws=42))