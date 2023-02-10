# Copied from here: https://github.com/cwerner/covid19/blob/master/app.py
import altair as alt

SCALE = alt.Scale(type='linear')
SCALE = alt.Scale(type='log', domain=[10, int(max(confirmed.confirmed))], clamp=True)


c2 = alt.Chart(confirmed.reset_index()).properties(height=150).mark_line().encode(
x=alt.X("date:T", title="Date"),
y=alt.Y("confirmed:Q", title="Cases", scale=SCALE),
color=alt.Color('country:N', title="Country")
)


c4 = alt.Chart(per100k.reset_index()).properties(width=75).mark_bar().encode(
  x=alt.X("per100k:Q", title="Cases per 100k inhabitants"),
  y=alt.Y("country:N", title="Countries", sort=None),
  color=alt.Color('country:N', title="Country"),
  tooltip=[alt.Tooltip('country:N', title='Country'), 
           alt.Tooltip('per100k:Q', title='Cases per 100k'),
           alt.Tooltip('inhabitants:Q', title='Inhabitants [mio]')]
)

st.altair_chart(alt.hconcat(c4, alt.vconcat(c2, c3)), use_container_width=True)


c = alt.Chart(dfm.reset_index()).mark_bar().properties(height=200).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("sum(value):Q", title="Cases", scale=alt.Scale(type='linear')),
            color=alt.Color('variable:N', title="Category", scale=SCALE), #, sort=alt.EncodingSortField('value', order='ascending')),
            order='order'
        )
st.altair_chart(c, use_container_width=True)
