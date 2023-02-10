RegEx remainder:
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line

RegEx: remove one or more blanks by just one blank
latexString = re.sub(' +', ' ', latexString)


# Find unique ids in a string
json_data = '''[{'S': 'kjahsdfbsl', 'Data': b'{"thingName":"yd_de7b49e6b0a7736b","rTs":158842}},''' + \
'''{'S': 'kjahsddslkjhg', 'Data': b'{"thingName":"yd_aa7b49e6b0a7736b","rTS":158842}}'''
current_RE = '''thingName\":\"(\w+)\"'''
regex  = re.compile(current_RE)
unq_devices = set([m.group(1) for m in re.finditer(regex, json_data)])


# Find email
test_str = 'Hey, drop me a line at carlos.aguilar@ef.com'
lst = re.findall('\S+@\S+', test_str)
print(lst)


# Get the work between the slashes
html_url = 'https://github.com/efcloud/tf-tester/pull/2'
github_org   = 'efcloud'
github_repo = 'tf-tester'

current_RE = f'{github_org}/{github_repo}[/](\w+)[/]'
regex  = re.compile(current_RE)
result = regex.search(html_url).group(1)



# Get all the fields between brackets in a string and replace them
sqlCreateTable = cu.dataFrameToCreateTableSQL(bionicData);

# Use a regular expression to find any parts within quotes
for quoted_part in re.findall(r'\"(.+?)\"', sqlCreateTable):
    print(quoted_part)
    sqlCreateTable = sqlCreateTable.replace(quoted_part, quoted_part.replace(" ", "_"))

print(sqlCreateTable)


# Parse a list of regular expressions (use filter):
import re
thisRegEx = 'manolo-\\d{1}'
regEx  = re.compile(thisRegEx);
listIDs = ['manolo-12', 'manolo-8']
parsedList = list(filter(regEx.match, listIDs))


# Apply a regex to a DF
#dfPrices.Price will contain rows such as >> '[{'date': '2017-11-29', 'price': '£26.99'}]'
price = dfPrices.Price.astype(str).apply(lambda x: re.match(r'\[(.+?)\]', x)[1])



from string import digits
s = 'abc123def456ghi789zero0'
remove_digits = str.maketrans('', '', digits)
res = s.translate(remove_digits)



reBatchId  = '.*?_(\d+)';
regEx      = re.compile(reBatchId);



# strfind
# find all the ocurrences in the string
long_string = 'brought him to the vaccinations. The four vaccinations big needles'
word = 'vaccinations'
import re
[m.start() for m in re.finditer(word, long_string)]


# Capture a string containing ISO8601 dates
this_regex = r'(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)Z'
regEx      = re.compile(this_regex)
regEx.match()



# Remove spaces using RegEx to clean a LaTeX table

import re

s = '''748.52 &                       654.52 &                     
420.01 &                     5741.67 &                     5972.15 &        
        604.69 &                560.79 &                 548.38 &                   
           560.37 &                      471.98 &       1720.06 & 1082.51 &          
            471.98 \\\\\n         RMSE &         1060.06 &         1403.48 &                    
              1154.81 &                     759.97 &                    14655.15 &               
                   18063.07 &           
                  1306.60 &               1183.18 &   
                            852.32 &                     1202.04 &                      873.38 &      
                             2620.21 & 1922.16 &           873.38 \\\\\n    meanError &          120.00 &     
                                 -511.68 &                        -2.57 &                    -201.14 &              
                                       -5162.12 &                    -5353.96 &                
           310.11 &               -299.31'''

clean_table = re.sub(r'(\s+)[\d|-]', ' ', s)
print('Remove spaces using RegEx:\n',  clean_table )





import re 
s = "[twice a day] This is a sentence. (once a day)"
re.sub("[\[].*?[\]]", "", s)



# Some more examples
this_regex = r'(\d+).(\d+).(\d+).(\d+).(\d+)'
regEx      = re.compile(this_regex)
rex_match  = regEx.findall(current_string)


this_regex = r'(\d+)[\.|_]'
regEx      = re.compile(this_regex)
rex_match  = regEx.findall(current_string)