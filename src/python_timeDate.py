import pandas as pd
import numpy as np
import random
from datetime import datetime, relativedelta

r = random.Random(10)
def generate_random_date_in_last_yeMar():
    return datetime.now() - relativedelta(years=0,days=365*random.random())


# Print out current time
time_format='%d-%m-%Y T%H:%M:%S'
datetime.now().strftime(time_format)


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

