trend_decompositions.py

# Some trend decompositions

# Compute the analytic signal, using the Hilbert transform.
from scipy.signal import hilbert
analytic_signal = hilbert(df_store.sales.values)
amplitude_envelope = np.abs(analytic_signal)

# Season-Trend decomposition using LOESS.
from statsmodels.tsa.seasonal import STL
sales_decomposition_LOESS = STL(df_store.sales, period=14).fit()

from statsmodels.tsa.seasonal import seasonal_decompose
sales_decomposition = seasonal_decompose(df_store.sales, model='additive', freq=14)





# Plot one store
f, ax = plt.subplots(1,1,figsize=(fig_w*1.5, fig_h/1.5))

ax.plot(x_axis, df_store.sales, label=f'Sales {current_store}', 
        color=def_colours[idx_store], linewidth=2, alpha=0.95)

ax.plot(x_axis, amplitude_envelope, label=f'Envelope {current_store}', 
        color=def_colours[idx_store+1], linewidth=1.5, alpha=0.95)

ax.plot(x_axis, sales_decomposition_LOESS.trend, label=f'LOESS trend', 
        color=def_colours[idx_store+2], linewidth=1.5, alpha=0.95)

ax.plot(x_axis, sales_decomposition.trend, label=f'Seasonal trend', 
        color=def_colours[idx_store+3], linewidth=1.5, alpha=0.95)


plt.legend()
plt.xlabel('dates')
plt.ylabel(f'Store sales for {sku_name} ')
plt.grid(True)
plt.show()






# Season-Trend decomposition using LOESS.
from statsmodels.tsa.seasonal import STL

def decompose_signal(input_signal, period_in_days=14, minimum_heartbeat=0.85):
    sales_decomposition_LOESS = STL(df_store.sales, period=period_in_days).fit()
    seasonality_flag = sales_decomposition_LOESS.trend>0.85
    return {'seasonality_flag': seasonality_flag,
        'trend': sales_decomposition_LOESS.trend,
        'seasonal': sales_decomposition_LOESS.seasonal
        'residual': sales_decomposition_LOESS.res}