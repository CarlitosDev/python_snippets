Date formats:
# Must be another way without calling datetime...
from datetime import datetime
datetime.strptime('02-07-2017', '%d-%m-%Y').date()

From timestamp to string:
	 stocks.index[-1].strftime('%Y-%m-%d')

From timestamp to string:
	dateAsStr = dfTest['date'].dt.strftime('%d_%m_%Y');

Produce a timestamp as string (saving files)
from datetime import datetime
datetime.now().strftime('%d_%m_%Y')




# Cast list of integers to string
ls = [i for i in range(10)]
# in one string separated by commas
ls_str = ','.join(map(str, ls))