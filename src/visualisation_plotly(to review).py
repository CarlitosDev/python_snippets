# Create fancy table


import plotly.plotly as py
import plotly.figure_factory as ff

data_matrix = [['Name', 'Equation'],
               ['Pythagorean Theorem', '$a^{2}+b^{2}=c^{2}$'],
               ['Euler\'s Formula', '$F-E+V=2$'],
               ['The Origin of Complex Numbers', '$i^{2}=-1$'],
               ['Einstein\'s Theory of Relativity', '$E=m c^{2}$']]

table = ff.create_table(data_matrix)
py.iplot(table, filename='latex_table')



# From PD
import plotly.plotly as py
import plotly.figure_factory as ff

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
df_sample = df[100:120]

table = ff.create_table(df_sample)
py.iplot(table, filename='pandas_table')