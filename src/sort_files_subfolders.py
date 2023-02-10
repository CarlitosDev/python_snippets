import os
from datetime import datetime
from shutil import copyfile, move

def sort_files_in_folder_yme(thisPath):
    '''
    Organise all files from folder and subfolders as basefolder/sorted/year/month/extension
    '''
    for root, dirs, files in os.walk(thisPath):
        for f in files:
          filePath = os.path.join(root, f)
          if os.path.isfile(filePath) and '.DS_Store' not in filePath and 'sort_files.py' not in filePath:
              file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
              _, current_extension = os.path.splitext(filePath)
              folderName = os.path.join(thisPath, 'sorted', str(file_ct.year), \
                      file_ct.strftime("%B"), current_extension.replace('.',''))
              if not os.path.exists(folderName):
                  os.makedirs(folderName)
              newFilePath = os.path.join(folderName, f)    
              move(filePath, newFilePath)
              print(f'Moving {newFilePath}...')


def sort_files_in_folder_ym(thisPath):
    '''
    
    Organise all files from folder and subfolders as basefolder/sorted/year/month

    '''
    for root, dirs, files in os.walk(thisPath):
        for f in files:
          filePath = os.path.join(root, f)
          if os.path.isfile(filePath) and '.DS_Store' not in filePath and 'sort_files.py' not in filePath:
              file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
              _, current_extension = os.path.splitext(filePath)
              folderName = os.path.join(thisPath, 'sorted', str(file_ct.year), file_ct.strftime("%B"))
              if not os.path.exists(folderName):
                  os.makedirs(folderName)
              newFilePath = os.path.join(folderName, f)    
              move(filePath, newFilePath)
              print(f'Moving {newFilePath}...')


def sort_files_in_folder_year_ext(thisPath):
    '''
    Organise all files from folder and subfolders as basefolder/sorted/year/extension
    '''
    for root, dirs, files in os.walk(thisPath):
        for f in files:
          filePath = os.path.join(root, f)
          if os.path.isfile(filePath) and '.DS_Store' not in filePath and 'sort_files.py' not in filePath:
              file_ct = datetime.fromtimestamp(os.stat(filePath).st_birthtime)
              _, current_extension = os.path.splitext(filePath)
              folderName = os.path.join(thisPath, 'sorted', str(file_ct.year), \
                current_extension.replace('.',''))
              if not os.path.exists(folderName):
                  os.makedirs(folderName)
              newFilePath = os.path.join(folderName, f)    
              move(filePath, newFilePath)
              print(f'Moving {newFilePath}...')


if __name__ == "__main__":
  sortIncludingMonth = True
  thisPath = os.path.dirname(os.path.realpath(__file__))
  print(f'This is the path {thisPath}...')
  if sortIncludingMonth:
    sort_files_in_folder_ym(thisPath)
  else:
    sort_files_in_folder_year_ext(thisPath)