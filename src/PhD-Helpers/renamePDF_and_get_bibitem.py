#pip install PyPDF2, scholarly

# importing required modules
import PyPDF2
import os
from os import listdir, rename
from os.path import isfile, join, dirname
from scholarly import scholarly
import requests
import json

def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)

path2PDFs      = os.path.dirname(os.path.realpath(__file__))
folderContents = listdir(path2PDFs)
pdfExt         = 'pdf';
pdfFiles       = [f for f in folderContents if isfile(join(path2PDFs, f)) and f.endswith(pdfExt)]


currentFile   = pdfFiles[0];
for currentFile in pdfFiles:

    # creating a pdf file object
    filePath   = join(path2PDFs, currentFile);
    pdfFileObj = open(filePath, 'rb');
    
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj);
    
    # printing number of pages in pdf file
    print(pdfReader.numPages);

    # extract PDF title
    pdfTitle = pdfReader.getDocumentInfo().title;
    
    # extracting text from page
    print(pdfTitle)
    
    # closing the pdf file object
    pdfFileObj.close()

    # lenght 
    if pdfTitle:
        titleLen = len(pdfTitle);
        maxLen   = 40;
        if titleLen>maxLen:
            pdfTitle = pdfTitle[0:maxLen];

        # Go to Google to fetch the bib info
        try:
            pub_info = scholarly.search_single_pub(pdfTitle)
            s = requests.get(pub_info.url_scholarbib)
            if s.ok:
                thisFile = join(path2PDFs, pdfTitle + '.bib')
                print(s.text)
                writeTextFile(s.text, thisFile)
        except:
            print(f'The thing failed...')

        # new file name
        pdfTitle    = pdfTitle + '.' + pdfExt;
        newFilePath = join(path2PDFs, pdfTitle);

        rename(filePath, newFilePath)

