flashtext_string_search.py


pip3 install flashtext



from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor(case_sensitive=False)




search_videos_with = ['car', 'airport']
keyword_processor.add_keywords_from_list(search_videos_with)


