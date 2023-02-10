'''

pip3 install QuantStats

'''



import quantstats as qs

# extend pandas functionality with metrics, etc.
# qs.extend_pandas()

# fetch the daily returns for a stock
symbol = 'ASC.L'
  stock = qs.utils.download_returns(symbol, period='2y')

# show sharpe ratio
qs.stats.sharpe(stock)

# qs.plots.snapshot(stock, title=f'{symbol} Performance', show=True)
# import matplotlib.pyplot as plt
# plt.show()


reportFile = '/Users/carlos.aguilar/Documents/tempRubbish/tester.html'
qs.reports.html(stock, output=reportFile, title=f'{symbol} Strategy Tearsheet')