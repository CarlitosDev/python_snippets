prepare_text_EF.py



# strip out HTML entities
re.compile(r'&[A-Za-z0-9#]+;')




var someString = 
json.content.replace(/<\/?([a-z][a-z0-9]*)\b[^>]*>/gi, '').
replace(/&#[0-9]+;t/gi,"").replace(/\[/g,"").replace(/\]/g,""); 



import re 
def cleanhtml(raw_html): 
	cleanr = re.compile('<.*?>') 
	cleantext = re.sub(cleanr)
	,  This method will demonstrate a way that we can remove html tags from a string using regex strings. 



import re 
TAG_RE = re.compile (r'< [^>]+>')	
def remove_tags(text): 
	return TAG_RE.sub('', text) 