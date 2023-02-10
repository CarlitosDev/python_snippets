import PyPDF2


def bookmark_dict(bookmark_list):
    result = {}
    for item in bookmark_list:
        if isinstance(item, list):
            # recursive call
            result.update(bookmark_dict(item))
        else:
            result[reader.getDestinationPageNumber(item)] = item.title
    return result


pdf_file = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/submissions/IEEE-November2k20/REVIEWS/papers/Retail forecasting_Research and practice.pdf'
reader = PyPDF2.PdfFileReader(pdf_file)

pdf_outlines = reader.getOutlines()

for k,v in pdf_outlines[0].items():
  print(k,v)


outliners = pdf_outlines[1]


bm = bookmark_dict(pdf_outlines)

for k,v in bm.items():
  print(k,v)




######

from typing import Dict

import fitz  # pip install pymupdf


def get_bookmarks(filepath: str) -> Dict[int, str]:
    # WARNING! One page can have multiple bookmarks!
    bookmarks = {}
    with fitz.open(filepath) as doc:
        toc = doc.getToC()  # [[lvl, title, page, …], …]
        for level, title, page in toc:
            bookmarks[page] = title
    return bookmarks


bm = get_bookmarks(pdf_file)

for k,v in bm.items():
  print(k,v)



with fitz.open(pdf_file) as doc:
    toc = doc.getToC()  # [[lvl, title, page, …], …]
    for level, title, page in toc:
      print(level, title, page)

