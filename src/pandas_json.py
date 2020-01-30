
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


# there must be an elegant way of doing this...
def json_col_to_df(df, fieldname = 'record'):
    temp_str = ''
    for iField in df[fieldname]:
        temp_str += str(iField) + ','
    temp_str = '[ ' + temp_str[:-1] + ' ]'
    return json_normalize(json.loads(temp_str))
    # might be quicker to just do
    #return pd.read_json(temp_str, orient='records')