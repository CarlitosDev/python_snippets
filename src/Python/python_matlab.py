##
# Load mat files
# This one is working well for me with the Thai data - which are structs
##
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/data/Thailand-bpnsPrevSales/bpnsPrevSales_FROZEN_from_160101_to1161231.mat'

mat = loadmat(path, squeeze_me=False)
print(mat.keys())
data_var_name = 'promoData' # typically the last key
names = mat[data_var_name].dtype.names
ndata_C = {n: np.squeeze(mat[data_var_name][n][0][0]) for n in names}
df = pd.DataFrame.from_dict(ndata_C, orient='index').T
df = df.infer_objects()
df.base_product_number_std = df.base_product_number_std.apply(lambda x: x[0])
df.calendar_date = df.calendar_date.apply(lambda x: x[0])
df.calendar_date = pd.to_datetime(df.calendar_date.astype(str), format='%Y-%m-%d')

df.iloc[21]

unique_items = df.base_product_number_std.unique()
num_items = len(unique_items)
num_promos = df.shape[0]



# Matlab and Python
pip3 install hdf5storage
import hdf5storage
import os
out = hdf5storage.loadmat('/Users/carlos.aguilar/Google Drive/order/BioEngineering Part/Data (ungitted)/Auriculas/#2/Carto/EG.mat')
import pandas as pd
df = pd.DataFrame.from_dict(out, orient='index')


# Get the info from the file
import os
import pandas as pd
import json
out = hdf5storage.loadmat('/Users/carlos.aguilar/Google Drive/order/BioEngineering Part/Data (ungitted)/Auriculas/#2/Carto/EG.mat')
df = pd.DataFrame.from_dict(out, orient='index')
print(df.ix[0][0])

# Another file
baseFolder  = '/Users/carlos.aguilar/Google Drive/order/BioEngineering Part/Data (ungitted)/Arrixaca_2k18/Case_2'
currentFile =  'ForGUI_1-VI.mat'
currentPath = os.path.join(baseFolder, currentFile)
out = hdf5storage.loadmat(currentPath)

creationInfo = (out['__header__']).decode("utf-8")
print(creationInfo)


statinfo = os.stat(currentPath)
sizeMB   = round(statinfo.st_size/(10**6))

fileComments = '''From Arrixaca. 
This file includes EGM data (both raw and processed), the mesh data and the car data.
The file was generated with prepareArrixacaDataForGUI.m'''
folderFromWD = baseFolder.replace('/Users/carlos.aguilar/Google Drive/order', '.')

fileDirections = '''This data will get loaded as a table'''

df = pd.DataFrame({'creationInfo': creationInfo, \
				   'folder': folderFromWD,
				   'filename': currentFile,
				   'comments': fileComments,
				   'directions': fileDirections,
				   'sizeMB': sizeMB}, 
				   index=[0])

df.to_json(orient='records')

jsonFName = currentPath.replace('.mat', '.json')
with open(jsonFName, 'w') as outfile:
    json.dump(data, outfile)



# ------------------------------------
# Example of a data catalog
# ------------------------------------
baseFolder  = '/Users/carlos.aguilar/Google Drive/order/BioEngineering Part/Data (ungitted)/Arrixaca_2k18/Case_5'
filesInFolder = os.listdir(baseFolder);
matFiles      = [f for f in filesInFolder if f.endswith('.mat')]
for currentFile in matFiles:
    currentPath = os.path.join(baseFolder, currentFile)
    out = hdf5storage.loadmat(currentPath)

    creationInfo = (out['__header__']).decode("utf-8")
    print(creationInfo)


    statinfo = os.stat(currentPath)
    sizeMB   = round(statinfo.st_size/(10**6))

    fileComments = '''From Arrixaca (high density VT). 
    This file includes EGM data (both raw and processed), the mesh data and the car data.
    This high level file was generated with prepareArrixacaDataForGUI.m that processes, filters out and produces some descriptors.
    The raw-est data come from PentaRay studies that are exported as zip files.
    To get that data into Matlab (useful for future cases), please follow the steps as per readArrixacaCasesFromZip.m'''
    folderFromWD = baseFolder.replace('/Users/carlos.aguilar/Google Drive/order', '.')

    fileDirections = '''This data will get loaded as a table'''

    df = pd.DataFrame({'creationInfo': creationInfo, \
                    'folder': folderFromWD,
                    'filename': currentFile,
                    'comments': fileComments,
                    'directions': fileDirections,
                    'sizeMB': sizeMB}, 
                    index=[0])


    jsonFName = currentPath.replace('.mat', '.json')
    with open(jsonFName, 'w') as outfile:
        json.dump(df.to_json(orient='records'), outfile)
    print('Done!')

    filesInfo.append(df)


# ------------------------------------
# Example for reading the data from a Kaggle competition
# where the data was hosted in a .mat file
# ------------------------------------
def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata





# ------------------------------------
# Save data into Matlab files
# ------------------------------------
from scipy.io import savemat
import numpy as np
a = np.arange(20)
mdic = {"a": a, "label": "experiment"}
mdic



savemat("matlab_matrix.mat", mdic)