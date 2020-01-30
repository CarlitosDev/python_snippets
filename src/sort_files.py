import os
from datetime import datetime
from shutil import copyfile, move

def sort_files_in_folder(thisPath):
    '''
    Sort files in a folder by sorted/year/month/extension
    '''
    files = []
    for f in os.listdir(thisPath):
        filePath = os.path.join(thisPath, f)
        if os.path.isfile(filePath) and '.DS_Store' not in filePath and 'sort_files.py' not in filePath:
            files.append(f)
            file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
            _, current_extension = os.path.splitext(filePath)
            folderName = os.path.join(thisPath, 'sorted', str(file_ct.year), \
                    file_ct.strftime("%B"), current_extension.replace('.',''))
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            newFilePath = os.path.join(folderName, f)    
            move(filePath, newFilePath)
            print(f'Copying {newFilePath}...')




#
if __name__ == "__main__":
  thisPath = os.path.dirname(os.path.realpath(__file__))
  print(f'This is the path {thisPath}...')
  sort_files_in_folder(thisPath)