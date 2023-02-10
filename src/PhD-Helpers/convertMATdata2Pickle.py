'''

  Convert mat files into Pandas Dataframes 
  and save them as pickle files

'''

import pandas as pd
import numpy as np
import scipy.io as sio
import os
import getpass
import pickle



# Read
def readPickleFile(filePath):
    with open(filePath, 'rb') as fId:
        pickleData = pickle.load(fId)
    return pickleData

# Write
def dataFrameToPickle(df, currentFile):
    with open(currentFile, 'wb') as f:
        pickle.dump(df, f)

# Process
def mat2DF(dataFile, pickleFile):

  print(f'Reading {dataFile}...')

  matData   = sio.loadmat(dataFile)

  data_keys = [*matData.keys()]
  temp_data = matData[data_keys[-1]]
  
  if temp_data != []:
    columns = list(temp_data.dtype.names)
    # This is crazy...
    df_list = []
    for idx, iCol in enumerate(columns):
      df_list.append(pd.DataFrame(temp_data[0][0][idx], columns=[iCol]))

    df = pd.concat(df_list, axis = 1)

    nR, nC = df.shape
    objTypes = df.select_dtypes(include=['object']).keys().tolist()

    if objTypes != []:
      for i_col in objTypes:
        print(f'Fixing {i_col}...')
        df[i_col] = df[i_col].apply(lambda x: x[0])
    
    print(f'writing {pickleFile}...')
    dataFrameToPickle(df, pickleFile)



if __name__ == "__main__":

  machine_name = getpass.getuser()
  baseFolder  = f'/Users/{machine_name}/Google Drive/order/Machine Learning Part/data'
  baseFolder  = f'/Users/{machine_name}/Google Drive/order/Machine Learning Part/data/data From the UK (by folder)'
  baseFolder = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/results/produce'
  baseFolder = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/Sergio IVI/IVI/IEEE paper data'
  for root, dirs, files in os.walk(baseFolder):
    for _file in files:
      dataFile     = os.path.join(root, _file)
      pickleFile   = dataFile.replace('.mat', '.pickle')
      pickleExists = os.path.exists(pickleFile)
      if '.mat' in _file and not pickleExists:
        mat2DF(dataFile, pickleFile)