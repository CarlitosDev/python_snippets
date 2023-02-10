import pytz
import pandas as pd


countryList = ['Japan','Morocco','Russia','Saudi Arabia','Taiwan']
ISOCountry2CharsCode = ['jp', 'ma', 'ru', 'sa', 'tw']
expectedTimes = ['2018-02-09 20:30:00+09:00',
'2018-02-09 12:30:00+01:00','2018-02-09 13:30:00+02:00',
'2018-02-09 14:30:00+03:00','2018-02-09 19:30:00+08:00']
 
df = pd.DataFrame(
    {'start_date': pd.Timestamp('09/02/2018  06:30:00'),
    'country': pd.Categorical(countryList),
    'local_start_date': expectedTimes,
    'ISOCountry2CharsCode': ISOCountry2CharsCode
    })


# Play around with time zones
# i - start_date must reflect that is 'US/Eastern' time

dw_timezone = pytz.timezone('America/Chicago')
df['start_local_time'] = df['start_date'].dt.tz_localize(dw_timezone, ambiguous='NaT')

for idx_A, iCountry in enumerate(ISOCountry2CharsCode):
    student_timezone = pytz.country_timezones(iCountry)
    df.loc[idx_A, 'start_local_time'] = \
            df.loc[idx_A, 'start_local_time'].tz_convert(student_timezone[0])


#df['start_time_UTC'] = df['start_local_time'].dt.tz_localize('UTC')


# iso8601 (timezone included)
import datetime
datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


# Example of ISO8601
# 2019-08-26T09:00:00Z
# midnight UTC (00:00:00Z)



# Set a string as a timezone and localize it 
pd.to_datetime('02-09-2019 05:00:00', format='%d-%m-%Y %H:%M:%S').tz_localize('America/New_York')



# Datetime localised
idx_valid = df_timeslots_adhoc.slot_start < datetime.now(tz=pytz.UTC)


# Localise UTC
# Convert to timestamp
import pytz
import datetime
from dateutil import tz
a = datetime.datetime(2018, 9, 25, 10, 12, 12, 183110, tzinfo=pytz.FixedOffset(60))
b = a.astimezone(tz.tzutc())
# to timestamp
timestamp = datetime.datetime.timestamp(b)


fcn_dt = lambda xt: datetime.datetime.timestamp(xt.astimezone(tz.tzutc()))