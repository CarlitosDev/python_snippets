'''
some_streamlit_decorators.py
'''


import streamlit as st
@st.cache(allow_output_mutation=True,
          hash_funcs={ GradREC: lambda _ : _.fclip.model_name })




# To show a gif
gif_mini_path = '....gif'
with open(gif_mini_path, "rb") as f:
    contents = f.read()
    data_url = base64.b64encode(contents).decode("utf-8")

exp_func = st.expander("Traversal Function")
exp_func.markdown(
    """
    <img src="data:image/gif;base64,{}" alt="cat gif" witdth=100 height=auto>
    &nbsp  
    &nbsp  
    """.format(data_url),
    unsafe_allow_html=True)


# to show an app with more than one page
# here intro.py is a Streamlit app that can run alone
# def app():
# 	st.write("""....""")
#    

import streamlit as st
import intro

PAGES = {
    "Intro": intro,
    "Intro2": intro,
}

page = st.sidebar.selectbox("", list(PAGES.keys()))

if page == "Intro":
    PAGES[page].app()
else:
    PAGES[page].app()