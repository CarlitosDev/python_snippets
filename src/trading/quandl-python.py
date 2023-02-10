'''
	quandl-python.py

	source ~/.bash_profile && pip3 install quandl

	https://github.com/quandl/quandl-python
'''



import quandl
# quandl.ApiConfig.api_key = 'ezvbHoySv-kEVjyzyDEC'

quandl.ApiConfig.verify_ssl = False

import quandl
# quandl.save_key('ezvbHoySv-kEVjyzyDEC')
print(quandl.ApiConfig.api_key)


import quandl
quandl.read_key()
print()


data = quandl.get_table('ZACKS/FC', ticker='AAPL')
# data = quandl.get_table('XLON')
data2 = quandl.get("BOE/RPAVAAULCBK", authtoken=quandl.ApiConfig.api_key)
# data = quandl.get_table('XLON', ticker='SBRY')
# data = quandl.get_table('ZACKS/FC', ticker='AAPL', qopts={'columns': ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']})
data3 = quandl.get("MULTPL/SP500_DIV_YIELD_MONTH", authtoken=quandl.ApiConfig.api_key)


# London GOLD price
data_gold = quandl.get("LBMA/GOLD", authtoken=quandl.ApiConfig.api_key)

# London SILVER price
data_silver = quandl.get("LBMA/SILVER", authtoken=quandl.ApiConfig.api_key)


# Radix to Bitcoin
data_bitcoin = quandl.get("BITFINEX/XRDBTC", authtoken=quandl.ApiConfig.api_key)

