# Multiline string without \n
description = ('SQL query running against Apollo.dbo.DimStudent '
'to get the profile info for the 500 students')

# Equivalent to Matlab's extractBetween
import re

def extractBetween(str, startStr, endStr):
  str_search = re.compile(f'{startStr}(.*?){endStr}')
  result = re.search(str_search, str)
  if result != None:
    result = result.group(1)
  return result


# Covert to snake_case
# pip3 install stringcase
import stringcase
a = 'StudentLevelTestProgress_id'
stringcase.snakecase(a)
# 'student_level_test_progress_id'

b = 'CourseAlterKey'
stringcase.snakecase(b)
# 'course_alter_key'

stringcase.capitalcase('atun de_a')
#'Atun de_a'

stringcase.snakecase(a)
#'student_level_test_progress_id'

stringcase.pascalcase(a)
#'StudentLevelTestProgressId'

stringcase.sentencecase(a)
#'Student level test progress id'

# First letter capitalised 
'ozu'.capitalize()

# to keep the camelCases
my_capitalise = lambda s: s[0].upper() + s[1::]
my_capitalise('ozuPisha')



# Count the number of occurrences
s = '0.8.2.3.3.How_often_do_you_ask_for_help'
s.count('.')