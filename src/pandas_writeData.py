Save a DF compressed:
dfTest[['id','unit_sales']].to_csv('ma8dwof.csv.gz', index=False, float_format='%.3f', compression='gzip')



Save DF to Excel:
fName     = 'winequality-red.xlsx';
xlsRoot   = '/Users/carlosAguilar/Documents/PythonDev/Coding/data for testing';
xlsFile   = os.path.join(dataRoot, fName)
xlsWriter = pd.ExcelWriter(xlsFile)
df.to_excel(xlsWriter, 'Red')
xlsWriter.save();



# Save to Excel in several sheets
with pd.ExcelWriter(filePath) as writer:
  for deviceId in listOfDeviceId:
    idx_device = all_devices.thingName.str.contains(deviceId)
    df = all_devices[idx_device].copy()
    df.sort_values(['startTimestamp'], inplace=True)
    df.to_excel(writer, sheet_name=deviceId)