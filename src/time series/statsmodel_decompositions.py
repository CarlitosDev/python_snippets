# Seasonal decompositions with statsmodel
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


df = pd.DataFrame.from_dict({'date': {0: pd.Timestamp('2015-04-25 00:00:00'), 1: pd.Timestamp('2015-04-26 00:00:00'), 2: pd.Timestamp('2015-04-27 00:00:00'),
	3: pd.Timestamp('2015-04-28 00:00:00'), 4: pd.Timestamp('2015-04-29 00:00:00'), 5: pd.Timestamp('2015-04-30 00:00:00'), 6: pd.Timestamp('2015-05-01 00:00:00'),
	7: pd.Timestamp('2015-05-02 00:00:00'), 8: pd.Timestamp('2015-05-03 00:00:00'), 9: pd.Timestamp('2015-05-04 00:00:00'), 10: pd.Timestamp('2015-05-05 00:00:00'),
	11: pd.Timestamp('2015-05-06 00:00:00'), 12: pd.Timestamp('2015-05-07 00:00:00'), 13: pd.Timestamp('2015-05-08 00:00:00'), 14: pd.Timestamp('2015-05-09 00:00:00'),
	15: pd.Timestamp('2015-05-10 00:00:00'), 16: pd.Timestamp('2015-05-11 00:00:00'), 17: pd.Timestamp('2015-05-12 00:00:00'), 18: pd.Timestamp('2015-05-13 00:00:00'),
	19: pd.Timestamp('2015-05-14 00:00:00'), 20: pd.Timestamp('2015-05-15 00:00:00'), 21: pd.Timestamp('2015-05-16 00:00:00'), 22: pd.Timestamp('2015-05-17 00:00:00'),
	23: pd.Timestamp('2015-05-18 00:00:00'), 24: pd.Timestamp('2015-05-19 00:00:00'), 25: pd.Timestamp('2015-05-20 00:00:00'), 26: pd.Timestamp('2015-05-21 00:00:00'),
	27: pd.Timestamp('2015-05-22 00:00:00'), 28: pd.Timestamp('2015-05-23 00:00:00'), 29: pd.Timestamp('2015-05-24 00:00:00'), 30: pd.Timestamp('2015-05-25 00:00:00'),
	31: pd.Timestamp('2015-05-26 00:00:00'), 32: pd.Timestamp('2015-05-27 00:00:00'), 33: pd.Timestamp('2015-05-28 00:00:00'), 34: pd.Timestamp('2015-05-29 00:00:00'),
	35: pd.Timestamp('2015-05-30 00:00:00'), 36: pd.Timestamp('2015-05-31 00:00:00'), 37: pd.Timestamp('2015-06-01 00:00:00'), 38: pd.Timestamp('2015-06-02 00:00:00'),
	39: pd.Timestamp('2015-06-03 00:00:00'), 40: pd.Timestamp('2015-06-04 00:00:00'), 41: pd.Timestamp('2015-06-05 00:00:00'), 42: pd.Timestamp('2015-06-06 00:00:00'),
	43: pd.Timestamp('2015-06-07 00:00:00'), 44: pd.Timestamp('2015-06-08 00:00:00'), 45: pd.Timestamp('2015-06-09 00:00:00'), 46: pd.Timestamp('2015-06-10 00:00:00'),
	47: pd.Timestamp('2015-06-11 00:00:00'), 48: pd.Timestamp('2015-06-12 00:00:00'), 49: pd.Timestamp('2015-06-13 00:00:00'), 50: pd.Timestamp('2015-06-14 00:00:00'),
	51: pd.Timestamp('2015-06-15 00:00:00'), 52: pd.Timestamp('2015-06-16 00:00:00'), 53: pd.Timestamp('2015-06-17 00:00:00'), 54: pd.Timestamp('2015-06-18 00:00:00'),
	55: pd.Timestamp('2015-06-19 00:00:00'), 56: pd.Timestamp('2015-06-20 00:00:00'), 57: pd.Timestamp('2015-06-21 00:00:00'), 58: pd.Timestamp('2015-06-22 00:00:00'),
	59: pd.Timestamp('2015-06-23 00:00:00'), 60: pd.Timestamp('2015-06-24 00:00:00'), 61: pd.Timestamp('2015-06-25 00:00:00'), 62: pd.Timestamp('2015-06-26 00:00:00'),
	63: pd.Timestamp('2015-06-27 00:00:00'), 64: pd.Timestamp('2015-06-28 00:00:00'), 65: pd.Timestamp('2015-06-29 00:00:00'), 66: pd.Timestamp('2015-06-30 00:00:00'),
	71: pd.Timestamp('2015-07-05 00:00:00'), 72: pd.Timestamp('2015-07-06 00:00:00'), 73: pd.Timestamp('2015-07-07 00:00:00'), 74: pd.Timestamp('2015-07-08 00:00:00'),
	75: pd.Timestamp('2015-07-09 00:00:00'), 76: pd.Timestamp('2015-07-10 00:00:00'), 77: pd.Timestamp('2015-07-11 00:00:00'), 78: pd.Timestamp('2015-07-12 00:00:00'),
	79: pd.Timestamp('2015-07-13 00:00:00'), 80: pd.Timestamp('2015-07-14 00:00:00'), 81: pd.Timestamp('2015-07-15 00:00:00'), 82: pd.Timestamp('2015-07-16 00:00:00'),
	83: pd.Timestamp('2015-07-17 00:00:00'), 84: pd.Timestamp('2015-07-18 00:00:00'), 85: pd.Timestamp('2015-07-19 00:00:00'), 86: pd.Timestamp('2015-07-20 00:00:00'),
	87: pd.Timestamp('2015-07-21 00:00:00'), 88: pd.Timestamp('2015-07-22 00:00:00'), 89: pd.Timestamp('2015-07-23 00:00:00'), 90: pd.Timestamp('2015-07-24 00:00:00'),
	91: pd.Timestamp('2015-07-25 00:00:00'), 92: pd.Timestamp('2015-07-26 00:00:00'), 93: pd.Timestamp('2015-07-27 00:00:00'), 94: pd.Timestamp('2015-07-28 00:00:00'),
	95: pd.Timestamp('2015-07-29 00:00:00'), 96: pd.Timestamp('2015-07-30 00:00:00'), 97: pd.Timestamp('2015-07-31 00:00:00'), 98: pd.Timestamp('2015-08-01 00:00:00'),
	99: pd.Timestamp('2015-08-02 00:00:00')},
	'sales': {0: 3272.0, 1: 3220.0, 2: 2683.0, 3: 2474.0, 4: 2296.0, 5: 2577.0, 6: 3468.0, 7: 3425.0, 8: 3437.0, 9: 2692.0, 10: 2692.0, 11: 2840.0, 12: 2934.0, 13: 3071.0,
	14: 3653.0, 15: 2972.0, 16: 2547.0, 17: 2628.0, 18: 2786.0, 19: 2909.0, 20: 3356.0, 21: 3687.0, 22: 3358.0, 23: 2558.0, 24: 2689.0, 25: 2726.0, 26: 2853.0, 27: 3314.0,
	28: 3362.0, 29: 3059.0, 30: 2937.0, 31: 2427.0, 32: 2670.0, 33: 2591.0, 34: 2749.0, 35: 3507.0, 36: 3371.0, 37: 3080.0, 38: 3068.0, 39: 3485.0, 40: 3272.0, 41: 3590.0,
	42: 3883.0, 43: 3349.0, 44: 3042.0, 45: 3091.0, 46: 3028.0, 47: 3344.0, 48: 3664.0, 49: 3800.0, 50: 3410.0, 51: 3546.0, 52: 3035.0, 53: 2806.0, 54: 3280.0, 55: 3653.0,
	56: 3819.0, 57: 3299.0, 58: 2781.0, 59: 2853.0, 60: 2760.0, 61: 2962.0, 62: 3278.0, 63: 3612.0, 64: 3331.0, 65: 2851.0, 66: 2835.0, 67: 3158.0, 68: 3585.0, 69: 4389.0,
	70: 3276.0, 71: 3475.0, 72: 3155.0, 73: 3120.0, 74: 3362.0, 75: 3307.0, 76: 3483.0, 77: 3831.0, 78: 3493.0, 79: 2976.0, 80: 3243.0, 81: 2989.0, 82: 3160.0, 83: 3430.0,
	84: 3412.0, 85: 3328.0, 86: 2700.0, 87: 2758.0, 88: 2797.0, 89: 2872.0, 90: 3236.0, 91: 3158.0, 92: 3234.0, 93: 2690.0, 94: 2515.0, 95: 2624.0, 96: 2750.0, 97: 3369.0,
	98: 3685.0, 99: 3341.0}})

x_axis = df_A.date
category_total_sales = df.sales



decomposition = seasonal_decompose(category_total_sales, model='additive', period = 7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot gathered statistics
plt.figure(figsize=(fig_w*1.25, fig_h))
plt.subplot(411)
plt.plot(x_axis, category_total_sales, label='Original', color=def_colours[0])
plt.grid()
plt.legend(loc='best')
plt.subplot(412)
plt.plot(x_axis, trend, label='Trend', color=def_colours[1])
plt.legend(loc='best')
plt.subplot(413)
plt.plot(x_axis, seasonal,label='Seasonality', color=def_colours[2])
plt.legend(loc='best')
plt.subplot(414)
plt.plot(x_axis, residual, label='Residuals', color=def_colours[3])
plt.legend(loc='best')
plt.grid()
plt.tight_layout()