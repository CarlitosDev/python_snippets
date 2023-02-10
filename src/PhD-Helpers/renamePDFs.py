# pip3 install PyPDF2 --upgrade

# importing required modules
import PyPDF2
from os import listdir, rename
from os.path import isfile, join

path2PDFs      = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/Papers/rename me';

folderContents = listdir(path2PDFs);
pdfExt         = 'pdf';
pdfFiles       = [f for f in folderContents if isfile(join(path2PDFs, f)) and f.endswith(pdfExt)];


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

    if pdfTitle:
    
        # extracting text from page
        print(pdfTitle)
        
        # closing the pdf file object
        pdfFileObj.close()

        # lenght 
        titleLen = len(pdfTitle);
        maxLen   = 40;
        if titleLen>maxLen:
            pdfTitle = pdfTitle[0:maxLen];

        # new file name
        pdfTitle    = pdfTitle + '.' + pdfExt;
        newFilePath = join(path2PDFs, pdfTitle);

        rename(filePath, newFilePath)
        else:
            print('Empty title')
