"""

  Example of Python client calling Knowledge Graph Search API API
  from https://developers.google.com/knowledge-graph

  Knowledge_Graph_Search_API.py

"""
import os
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from utils.file_utils import printJSON


query = 'malcolm gladwell'


api_key = os.environ['GKG_API']
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    'limit': 5,
    'indent': True,
    'key': api_key,
}
url = service_url + '?' + urlencode(params)

response = json.loads(urlopen(url).read())
# for element in response['itemListElement']:
#   print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
printJSON(response)

'''
{
  "@context": {
    "detailedDescription": "goog:detailedDescription",
    "resultScore": "goog:resultScore",
    "EntitySearchResult": "goog:EntitySearchResult",
    "@vocab": "http://schema.org/",
    "goog": "http://schema.googleapis.com/",
    "kg": "http://g.co/kg"
  },
  "@type": "ItemList",
  "itemListElement": [
    {
      "result": {
        "detailedDescription": {
          "url": "https://en.wikipedia.org/wiki/Malcolm_Gladwell",
          "articleBody": "Malcolm Timothy Gladwell CM is an English-born Canadian journalist, author, and public speaker. He has been a staff writer for The New Yorker since 1996. ",
          "license": "https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License"
        },
        "@type": [
          "Person",
          "Thing"
        ],
        "name": "Malcolm Gladwell",
        "@id": "kg:/m/03ff3k",
        "description": "Canadian journalist"
      },
      "@type": "EntitySearchResult",
      "resultScore": 579.430419921875
    },
    {
      "resultScore": 24,
      "result": {
        "@type": [
          "Event",
          "Thing"
        ],
        "name": "Malcolm Gladwell",
        "@id": "kg:/g/11fnrwz14t"
      },
      "@type": "EntitySearchResult"
    }
  ]
}
'''