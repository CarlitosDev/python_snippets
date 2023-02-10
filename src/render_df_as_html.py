render_df_as_html.py
# from https://www.kaggle.com/code/lextoumbourou/feedback3-eda-hf-custom-trainer-sift

def print_texts(df):
    inner_html = ""
    for idx, row in df.iterrows():
        inner_html += f"""
        <td style="vertical-align:top; border-right: 1px solid #7accd8">
        <h3 style="text-align:left">{row.text_id}</h3>
        <hr>
            <dl>
                <dt>Cohesion</dt>
                <dd>{row.cohesion}</dd>
                
                <dt>Syntax</dt>
                <dd>{row.syntax}</dd>
                
                <dt>Vocabulary</dt>
                <dd>{row.vocabulary}</dd>
                
                <dt>Phraseology</dt>
                <dd>{row.phraseology}</dd>
                
                <dt>Grammar</dt>
                <dd>{row.grammar}</dd>
                
                <dt>Conventions</dt>
                <dd>{row.conventions}</dd>
            </dl>
            
            <hr>
            <p>
            {row.full_text[:400]}
            </p>
        </td>
        """

    display(HTML(f"""
    <table style="font-family: monospace;">
        <tr>
             {inner_html}
        </tr>
    </table>
    """))