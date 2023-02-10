'''
pip3 install streamlit --upgrade --ignore-installed

Bridging the gap from prototyping to app running

Principles:
	1- Embrace Python scripting
	2 - Treat widgets as variables, ie: x = st.slider('x')
	3 - Reuse data and computing, ie: st.cache()


streamlit run streamlit_intro.py

https://github.com/arvkevi/nba-roster-turnover/blob/master/roster_turnover.py



Some notes:

In st.beta_columns now you can put the columns anywhere in your app. 
In fact, by calling st.beta_columns inside a loop, you get a grid layout.


# Use the full page instead of a narrow central column
st.beta_set_page_config(layout="wide")
# Space out the maps so the first one is 2x the size of the other three
c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))


source ~/.bash_profile && python3 -m pip install st-annotated-text



'''

import streamlit as st
import numpy as np
import time
import pandas as pd

# import shapefile

st.empty()
my_bar = st.progress(0)
for i in range(100):
    my_bar.progress(i + 1)
    time.sleep(0.01)
n_elts = int(time.time() * 10) % 5 + 3
for i in range(n_elts):
    st.text("." * i)
st.write(n_elts)
for i in range(n_elts):
    st.text("." * i)
st.success("done")


# Show a DF
# Fake sales
num_samples = 500
num_features = 5
input_vars = [f'x_{idx}' for idx in range(1,num_features+1)]
input_data = np.random.rand(num_samples, num_features)
df = pd.DataFrame(input_data, columns=input_vars)

'data', df

st.markdown(
    """
This is a _markdown_ block...
```python
print('...and syntax hiliting works here, too')
```
"""
)



x = st.slider('Select a value')
st.write(x, 'squared is', x * x)



st.markdown('# Some plotly here')


import plotly.graph_objs as go

st.header("Chart with two lines")

trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
trace1 = go.Scatter(x=[1, 2, 3, 4], y=[16, 5, 11, 9])
data = [trace0, trace1]
st.write(data)


###

st.header("Matplotlib chart in Plotly")

import matplotlib.pyplot as plt

f = plt.figure()
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)

st.plotly_chart(f)


###

st.header("3D plot")

x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="markers",
    marker=dict(
        size=12,
        color=z,  # set color to an array/list of desired values
        colorscale="Viridis",  # choose a colorscale
        opacity=0.8,
    ),
)

data = [trace1]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)

st.write(fig)


###

st.header("Fancy density plot")

import plotly.figure_factory as ff

import numpy as np

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ["Group 1", "Group 2", "Group 3"]

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

# Plot!
st.plotly_chart(fig)



#
analysis = st.sidebar.selectbox("Choose Analysis", ["Overview", "By Country"])

multiselection = st.multiselect("Select countries:", countries, default=countries)
logscale = st.checkbox("Log scale", False)



##
st.header("Graphviz")
st.graphviz_chart('''
  digraph {
    rankdir=LR;
    a -> c;
    b -> c;
    c -> d;
    d -> e;
    d -> f;
    f -> g;
  }''')


###

# Plot a table from a DF
st.sidebar.table(
        pd.DataFrame.from_dict(
            wins_turnover_corr, orient="index", columns=["correlation"]
        ).round(2)
    )



# Show images




import streamlit as st
import SessionState


state = SessionState.get(chat_list=[])

name = st.sidebar.text_input("Name")
message = st.sidebar.text_area("Message")
if st.sidebar.button("Post chat message"):
    state.chat_list.append((name, message))

if len(state.chat_list) > 10:
    del (state.chat_list[0])

try:
    names, messages = zip(*state.chat_list)
    chat1 = dict(Name=names, Message=messages)
    st.table(chat1)
except ValueError:
    st.title("Enter your name and message into the sidebar, and post!")