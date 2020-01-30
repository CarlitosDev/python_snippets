# Try Modin out
import pandas as pd
import modin.pandas as mpd



# Note expectedTimes is deliberately left as strings
countryList = ['Japan','Morocco','Russia','Saudi Arabia','Taiwan']
expectedTimes = ['2018-02-09 20:30:00+09:00',
'2018-02-09 12:30:00+01:00','2018-02-09 13:30:00+02:00',
'2018-02-09 14:30:00+03:00','2018-02-09 19:30:00+08:00']
df = pd.DataFrame(
    {'start_date': pd.Timestamp('09/02/2018  06:30:00'),
    'country': pd.Categorical(countryList),
    'local_start_date': expectedTimes
    })




df_test = mpd.DataFrame(df)
df_test.describe()