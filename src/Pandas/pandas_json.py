
# This reads a nested dict and creates DF
from pandas.io.json import json_normalize
df_temp = json_normalize(df_bookingresponse.iloc[idx].bookings)

# To do it in one-go
df_adhoc_ext = df_adhoc[idx_one_range].reset_index()
df_json = pd.io.json.json_normalize(df_adhoc_ext.available_ranges.explode())
df_adhoc_ext = df_adhoc_ext.join(df_json)


# Differnet outcomes
df_json = pd.io.json.json_normalize(df_bookingresponse.iloc[idx].bookings)
df_json = pd.io.json.json_normalize(df_bookingresponse.bookings.explode())


# Dump to json (dates as 'iso')
df.to_json(date_format='iso')


# there must be an elegant way of doing this...
def json_col_to_df(df, fieldname = 'record'):
    temp_str = ''
    for iField in df[fieldname]:
        temp_str += str(iField) + ','
    temp_str = '[ ' + temp_str[:-1] + ' ]'
    return json_normalize(json.loads(temp_str))
    # might be quicker to just do
    #return pd.read_json(temp_str, orient='records')


def json_payload_to_df(json_dict):
    '''
        Turn a json payload (pass it as json.loads(payload))
        into a DF where rows are repeated
        
        Disclaimer: Still to work out some more edge cases

    '''
    expanded_cols = []
    for k,v in json_dict.items():
        if isinstance(v, list) & (v != []):
            if isinstance(v[0], str):
                expanded_cols.append(pd.DataFrame({k: v}))
            elif isinstance(v[0], dict):
                df_t = json_normalize(v)
                col_names = [k + '.' + iVar for iVar in df_t.columns.tolist()]
                df_t.columns = col_names
                expanded_cols.append(df_t)
        elif isinstance(v, dict):
            expanded_cols.append(json_normalize(v))
        elif (v != []):
            expanded_cols.append(pd.DataFrame({k: v}, index = [0]))
            
    return pd.concat(expanded_cols, axis=1)


# TO FINISH ONE DAY
'''
df = pd.io.json.json_normalize(subDict)
df = df.infer_objects()
obj_columns = df.select_dtypes(include=['object'])

for iCol in df.select_dtypes(include=['object']):
    if any(df[iCol].str.contains('-|/|:')):
        try:
            result = parse(df[iCol][0], fuzzy_with_tokens=True)
        except ValueError:
            pass

contains('expert_model|challenger_model')

from dateutil.parser import parse


result = parse(df.iloc[-1].lesson_start_time, fuzzy_with_tokens=True)
result = parse("manolo", fuzzy_with_tokens=True)
'''