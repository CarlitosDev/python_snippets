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


# Add a chart to Excel
# define the workbook
workbook = writer.book
worksheet = writer.sheets[sheet_name]
# create a chart line object
chart = workbook.add_chart({'type': 'line'})
# configure the series of the chart from the spreadsheet
# using a list of values instead of category/value formulas:
#     [sheetname, first_row, first_col, last_row, last_col]
chart.add_series({
    'categories': [sheet_name, 1, 0, 3, 0],
    'values':     [sheet_name, 1, 1, 3, 1],
})
# configure the chart axes
chart.set_x_axis({'name': 'Index', 'position_axis': 'on_tick'})
chart.set_y_axis({'name': 'Value', 'major_gridlines': {'visible': False}})
# place the chart on the worksheet
worksheet.insert_chart('E2', chart)
# output the excel file
writer.save()


# To Snowflake
# https://docs.snowflake.com/en/user-guide/python-connector-api.html#module-snowflake-connector-pandas-tools
from snowflake.connector.pandas_tools import pd_writer
df.to_sql('customers', engine, index=False, method=pd_writer)
#https://github.com/snowflakedb/snowflake-connector-python