# Find files
import glob
outputBaseFolder = '~/Google Drive/order/Machine Learning Part/data/Walmart(M5)'
glob_pattern = os.path.join(os.path.expanduser(outputBaseFolder), '*', dept_id, '*', 'stores', '*.pickle')
store_level_files = glob.glob(glob_pattern)
for idx, this_file in enumerate(store_level_files):
    fhelp.print_every_n(f'Reading {this_file}', idx, 38)

# Get the username
import getpass
machine_name = getpass.getuser()

# Sort files in a folder by sorted/year/month/extension
def sort_files_in_folder(thisPath):
    '''
    Sort files in a folder by sorted/year/month/extension
    '''
    files = []
    for f in os.listdir(filePath):
        filePath = os.path.join(filePath, f)
        if os.path.isfile(filePath):
            files.append(f)
            file_ct = datetime.fromtimestamp(os.stat(filePath).st_ctime)
            _, current_extension = os.path.splitext(filePath)
            folderName = os.path.join(filePath, 'sorted', str(file_ct.year), file_ct.strftime("%B"), current_extension)
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            newFilePath = os.path.join(folderName, f)    
            copyfile(filePath, newFilePath)
            print(f'Copying {newFilePath}...')



# traverse root directory, and list directories as dirs and files as files
# naviage all folders
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)



# Correct the name of all subfolders in the likes of '.PNG', etc
from shutil import  move
baseFolder ='/Volumes/CarlosPictures/iPhone pics 9.12.18'
for root, dirs, files in os.walk(baseFolder):
  path = root.split(os.sep)
  baseName = os.path.basename(root)
  if '.' in baseName:
    newFilePath = root.replace(baseName, baseName.replace('.', ''))
    move(root, newFilePath)


# To use a temporary directory:
from tempfile import TemporaryDirectory
    with TemporaryDirectory() as local_target_location:
		# do stuff

# Find files with certain extension:
path2PDFs = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/Papers/papers';
pdfExt    = 'pdf'
folderContents = os.listdir(path2PDFs);
pdfFiles  = [f for f in folderContents if os.path.isfile(join(path2PDFs, f)) and f.endswith(pdfExt)]

# Sort files by creation time:
path2files      = '/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/' 
fullPath = [os.path.join(path2files, iFile) for iFile in os.listdir(path2files)]
fullPath.sort(key=os.path.getmtime, reverse=True)


# Get the filename and extention
name, ext = os.path.splitext()

# a bit better... (fileparts)
[fPath, fName] = os.path.split(this_image)
[file, ext] = os.path.splitext(fName)


# Implicitly, by opening files with the with statement. The close() method will be called 
# when the end of the with block is reached, even in the event of 
# abnormal termination (from an exception).
with open("data.txt") as in_file:
	data = in_file.read()

# Write file:
with open(fileToRun, 'w+') as in_file:
for thisFile in missingBackUp:
    srcPath = '''"s3://beamly-metrics-data-stage/''' + folderNameBMetrics[0] + '/' + thisFile + '"'
    thisCmd = cmdPrefix + srcPath + cmdPostFix + ';'
    in_file.write(thisCmd);



# The path module already has join, split, dirname, and basename functions
os.path.join(os.path.abspath('.'), 'utils')

# Join a path:
jsonPath = os.path.join(metaRoot, jsonFile);

# To get the parts (Matlab fileparts):
[fPath,fName] = os.path.split('/Volumes/Impressions/Impression_86132_Summary.pickle')



# Delete all the files with a particular extension within a folder:
rmCommand = '''find "{}" -name "{}"'''.format(metaTempFolder, '*.xml');
os.system(rmCommand)

# Split path into chunks:
import ntpath
thisFolder, thisFile = ntpath.split(subString)


# Use split to get parts of filenames:
fName = 'A.B.C'
partA = fName.split('.')[-3]


import sys
fid = sys.stderr
print("fatal error", file=fid)
but also:

fid = open('test.txt','w')
print("fatal error", file=fid)
fid.fclose()


# Make a folder if it doesn't exist:
dataFolder = 'xlsTickers';
if not os.path.exists(dataFolder):
    os.makedirs(dataFolder)


#  Walk a directory
os.walk creates an iterator with three items: the root directory, subdirectories, and files.

data_path  = os.path.join('..','data', 'raw', 'crimson_api')
files_list = list()
for root, dirs, files in os.walk(data_path):
    for file in files:
        temp_file = os.path.join(root, file)


# Get all the contents of a folder and subfolders
# glob library (only UNIX-based)
The glob module finds all the pathnames matching a specified pattern according to the 
rules used by the Unix shell, although results are returned in arbitrary order.

from glob import glob
patternA = '/Users/carlos.aguilar/Documents/Beamly/S2DS/Repos/Beamly2/data/raw/crimson_api/wild rose foundation/*.json'
a = glob(patternA)
# as we can use regex and wildcards, read all the contents of the folder + subfolders:
patternB = '/Users/carlos.aguilar/Documents/Beamly/S2DS/Repos/Beamly2/data/raw/crimson_api/**/*.json'
b = glob(patternB, recursive=True)