# pip3 install scholarly

from scholarly import scholarly
import requests
import json

# Retrieve the author's data, fill-in, and print
search_query = scholarly.search_author('Steven A Cholewiak')
author = next(search_query).fill()
print(author)

# Print the titles of the author's publications
print([pub.bib['title'] for pub in author.publications])

# Take a closer look at the first publication
pub = author.publications[0].fill()
print(pub)

# Which papers cited that publication?
print([citation.bib['title'] for citation in pub.get_citedby()])


this_pub = 'forecasting promotional sales within the neighbourhood'
pub_info = scholarly.search_single_pub(this_pub)

# Request this one to get the citation in bibtex
pub_info.url_scholarbib
print(pub_info.url_scholarbib)

def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)

s = requests.get(pub_info.url_scholarbib)
if s.ok:
    print(s.text)
    writeTextFile(s.text, thisFile)
