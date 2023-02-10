'''

https://pypi.org/project/yfinance/


  There are two ways of doing stuff with YFinance.
  1 - The download method fetches the data and returns it as a DF
  2 - The ticker method is a class that stores plenty of information, including the above. AFAIK, the Ticker class cannot be initialised with downloaded data from (1)



  https://aroussi.com/ is the author of quantstats


'''

import yfinance as yf

data = yf.download(tickers='UBER', period='1d', interval='1m')



# to get data within 1 minute...
symbol = 'ASC.L'
data = yf.download(tickers=symbol, period='1d', interval='1m')



# to get data within 1 minute...
symbol = 'ASC.L'

# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

# YTD starts in Jan...
ticker_records = yf.download(tickers=symbol, period='ytd', interval='1d')
# Last year
ticker_records = yf.download(tickers=symbol, period='1y', interval='1d')


# 
ticker_records.info

# Using the ticker module
this_ticker = yf.Ticker(symbol)


import yfinance as yf

this_ticker = yf.Ticker("this_ticker")

# get stock info
ticker_info = this_ticker.info
import carlos_utils.file_utils as fu
fu.printJSON(ticker_info)

# get historical market data
# hist = this_ticker.history(period="max")

# show actions (dividends, splits)
this_ticker.actions

# show dividends
this_ticker.dividends

# show splits
this_ticker.splits

# show financials
this_ticker.financials
this_ticker.quarterly_financials

# show major holders
this_ticker.major_holders

# show institutional holders
this_ticker.institutional_holders

# show balance sheet
this_ticker.balance_sheet
this_ticker.quarterly_balance_sheet

# show cashflow
this_ticker.cashflow
this_ticker.quarterly_cashflow

# show earnings
this_ticker.earnings
this_ticker.quarterly_earnings

# show sustainability
this_ticker.sustainability

# show analysts recommendations
this_ticker.recommendations

# show next event (earnings, etc)
this_ticker.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
this_ticker.isin

# show options expirations
this_ticker.options

# show news
ticker_news = this_ticker.news
fu.printJSON(ticker_news[-1])

# get option chain for specific expiration
# opt = this_ticker.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts