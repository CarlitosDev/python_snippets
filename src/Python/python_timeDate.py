import pandas as pd
import numpy as np
import random
from datetime import datetime, relativedelta

# UNIX time. UNIX epochs
days_before = 3
end_time = int(time.time())*1000
days_before_in_ms = days_before*24*60*60*1000
start_time = end_time-days_before_in_ms
# also...
import time
epoch_in_ms = 1644667200000
epoch_time = int(epoch_in_ms/1000)
time_val = time.localtime(epoch_time)
print("Date in time_struct:", time_val)
time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_time))
# also...
from datetime import datetime
epoch_in_ms = 1647643980131
date_time = datetime.fromtimestamp(epoch_in_ms/1000)
time_format='%d-%m-%Y T%H:%M:%S'
date_str = date_time.strftime(time_format)



# time formatters
import time
# seconds to time in MM:SS
formatter_seconds_to_string = lambda secs: time.strftime('%M:%S', time.gmtime(secs))
# time HH:MM:SS to seconds
formatter_string_to_seconds = lambda hh_mm_ss: sum(x * int(t) for x, t in zip([1, 60, 3600], reversed(hh_mm_ss.split(":"))))



r = random.Random(10)
def generate_random_date_in_last_yeMar():
    return datetime.now() - relativedelta(years=0,days=365*random.random())


# Print out current time
time_format='%d-%m-%Y T%H:%M:%S'
datetime.now().strftime(time_format)

# From string to datetime and back to string (different format)
import datetime as dt
date_to_test = '22-02-2021'
date_time_obj = dt.datetime.strptime(date_to_test, '%d-%m-%Y')
date_prefix = date_time_obj.strftime('%Y/%m/%d')


# Dates differences:
from dateutil.relativedelta import relativedelta
endDT   = dt.datetime.today();
startDT = endDT - relativedelta(years=1);


# Generate date ranges and print them as iso8601
from dateutil.relativedelta import relativedelta
t  = datetime.datetime.now()
ts = pd.date_range(t, freq='W', periods=6)
ts.tz_localize('UTC')
for irow in ts:
  ts_begin = irow.replace(tzinfo=datetime.timezone.utc).isoformat()
  ts_end   = (irow + relativedelta(hours=2)).replace(tzinfo=datetime.timezone.utc).isoformat()
  print('{')
  print(f'''"begin": "{ts_begin}"''')
  print(f'''"end": "{ts_end}"''')
  print('},')





Produce a timestamp as string (saving files)
from datetime import datetime
datetime.now().strftime('%d_%m_%Y')

# Get the month name (as a string)
import datetime
datetime.datetime.now().strftime("%B")


Convert from standard date format to simple numbers in a dataframe
	import matplotlib.dates   as mdates	
	df['Date'] = df['Date'].map(mdates.date2num);


Timing processes (elapsed time) :
	startTime   = time.time();
	# do some magic
    endTime     = time.time();
    elapsedTime = endTime - startTime;
    print ('...completed {:.2f} sec!'.format(elapsedTime));


# Date difference in days
t2017a = date(2017, 5, 31)
t2017b = date(2017, 8, 31)
(t2017b-t2017a).days



# Dates differences:
import datetime as dt
from dateutil.relativedelta import relativedelta
endDT   = dt.datetime.today();
startDT = endDT - relativedelta(years=1);


# Dates windows:
import datetime as dt
from dateutil.relativedelta import relativedelta
endDT       = dt.datetime.today();
startDT     = endDT - relativedelta(months=3);
endDT_str   = endDT.strftime('%Y-%m-%d')
startDT_str = startDT.strftime('%Y-%m-%d')
print('From {} to {}'.format(startDT_str, endDT_str))


# get the following 16 days...
t2017 = date(2017, 5, 31)
pd.date_range(t2017, periods=16)


# into YYYYMMDDTHH format
print(dt.datetime.today().strftime('%Y%m%dT%H'))


# Elapsed time
import time
start = time.time()
end = time.time()
elapsed_time = end - start
print ('completed staging insert loops in {:.2f} sec!'.format(elapsed_time))


# Elapsed time from the command line...
python3 -m timeit '"-".join(str(n) for n in range(100))'

