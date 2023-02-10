'''
streamlit_with_Spacy.py


source ~/.bash_profile && python3 -m pip install spacy-streamlit --upgrade

Great documentation here
https://github.com/explosion/spacy-streamlit

'''

from spacy import displacy

html = displacy.render(
    doc, style="ent", options={"ents": label_select, "colors": colors}
)
style = "<style>mark.entity { display: inline-block }</style>"
st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)




import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
html = displacy.render(doc, style="dep")


options = {"compact": True, "color": "blue"}





annotated_text(
                    'This is normal text',
                    ("is", "verb", "#8ef"),
                    " some ",
                    ("annotated", "adj", "#ffaaaa", font_color),
                    ("text", "noun", "#afa", font_color),
                    " for those of ",
                    ("you", "pronoun", "#fea", font_color),
                    " who ",
                    ("like", "verb", "#8ef", font_color),
                    " this sort of ",
                    ("thing", "noun", "#afa", font_color),
                )