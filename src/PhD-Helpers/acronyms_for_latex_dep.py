
def acronym_to_list(this_acronym, this_meaning):
  mn = this_meaning.split()
  print('\\textbf{' + this_acronym + '} & ', end='')
  for idx, this_char in enumerate(list(this_acronym)):
    print('\\textbf{' + this_char + '}' + mn[idx][1::] + ' ', sep=' ', end='')
  print('\\\\')



# Place them all here so I can sort them

all_terms = {}

this_acronym = 'CPG'
this_meaning = 'consumer packaged goods'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'FMCG'
this_meaning = 'fast Moving Consumer Goods'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'BOGOF'
this_meaning = 'buy one get one free'
all_terms.update({this_acronym: this_meaning})







import collections
od = collections.OrderedDict(sorted(all_terms.items()))
for k,v in od.items():
  acronym_to_list(k, v)