'''
  pip3 install pybtex
'''

from pybtex.database.input import bibtex

bibtex_path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/contrastive explanations/bibtex/promotions.bib'

parser = bibtex.Parser()
bib_data = parser.parse_file(bibtex_path)
all_entries = sorted(bib_data.entries.keys(), key=lambda k: k.lower())



# merge all the bibs used during the thesis
import os 
from pybtex.database.input import bibtex

base_folder = '/Users/carlos.aguilar/Documents/temp_carlos'

filepath_A = os.path.join(base_folder, 'file1.bib')
parser = bibtex.Parser()
bib_data_A = parser.parse_file(filepath_A)

bib_data_A_entries = [*bib_data_A.entries.keys()]


filepath_B = os.path.join(base_folder, 'file2.bib')
parser_B = bibtex.Parser()
bib_data_B = parser_B.parse_file(filepath_B)

bib_data_B_entries = [*bib_data_B.entries.keys()]

to_add_from_B = [*set(bib_data_B_entries)-set(bib_data_A_entries)]

for this_entry in to_add_from_B:
  new_entry = bib_data_B.entries.get(this_entry)
  bib_data_A.add_entry(this_entry, new_entry)



filepath_C = os.path.join(base_folder, 'contrastive.bib')
parser_C = bibtex.Parser()
bib_data_C = parser_C.parse_file(filepath_C)

bib_data_C_entries = [*bib_data_C.entries.keys()]
bib_data_A_entries = [*bib_data_A.entries.keys()]
to_add_from_C = [*set(bib_data_C_entries)-set(bib_data_A_entries)]

for this_entry in to_add_from_C:
  new_entry = bib_data_C.entries.get(this_entry)
  bib_data_A.add_entry(this_entry, new_entry)


bib_data_A.to_file('test.bib')



##### Sort out the entries of the PhD

bibtex_path = '/Users/carlos.aguilar/Downloads/Doctoral Thesis (3)/bibliography.bib'

parser = bibtex.Parser()
bib_data = parser.parse_file(bibtex_path)
all_entries = sorted(bib_data.entries.keys(), key=lambda k: k.lower())

from pybtex.database import BibliographyData
sorted_bib_data = BibliographyData()

for this_entry in all_entries:
  new_entry = bib_data.entries.get(this_entry)
  sorted_bib_data.add_entry(this_entry, new_entry)

sorted_bibtex_path = '/Users/carlos.aguilar/Downloads/Doctoral Thesis (3)/bibliography_sorted.bib'
sorted_bib_data.to_file(sorted_bibtex_path)