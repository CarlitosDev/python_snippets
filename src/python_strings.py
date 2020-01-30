# Equivalent to Matlab's extractBetween
import re

def extractBetween(str, startStr, endStr):
  str_search = re.compile(f'{startStr}(.*?){endStr}')
  result = re.search(str_search, str)
  if result != None:
    result = result.group(1)
  return result