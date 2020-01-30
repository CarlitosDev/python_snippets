RegEx remainder:
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line

RegEx: remove one or more blanks by just one blank
latexString = re.sub(' +', ' ', latexString)


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
