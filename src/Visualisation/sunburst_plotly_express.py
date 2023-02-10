'''
sunburst_plotly_express.py
from https://plotly.com/python/sunburst-charts/
'''


import plotly.express as px
data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])

fig =px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
)
fig.show()




import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['day', 'sex'],
				values='total_bill', color='total_bill')
fig.show()






import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')

df.iloc[21]

fig = go.Figure(go.Sunburst(
        ids = df.ids,
        labels = df.labels,
        parents = df.parents))
fig.update_layout(uniformtext=dict(minsize=10, mode='hide'))
fig.show()

df.parents.unique()
df.labels.unique()



import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['day', 'time', 'smoker', 'sex'], values='total_bill')
fig.show()



import plotly.express as px
df = df_student.copy()

fig = px.sunburst(df, path=['day', 'time', 'smoker', 'sex'], values='total_bill')
fig.show()



for _item in df.signal_type.unique():
    print(f''''{_item}':'courseware',''')



###


action_type_mapper = {
'course_set':'courseware',
'enrollment':'courseware',
'activity':'courseware',
'step':'courseware',
'online_class': 'classes',
'lesson_completed':'courseware',
'unit_completed':'courseware',
'writing':'writing'
}

df_student = df.copy()
df_student['action_parent'] = df_student.signal_type.map(action_type_mapper)

import plotly.express as px
df = df_student.copy()
# fig = px.sunburst(df, path=['action_parent', 'platform', 'signal_type', 'content_type'], values='time_spent')
fig = px.sunburst(df, path=['action_parent', 'platform', 'signal_type', 'content_type'], values='time_spent')
fig.show()