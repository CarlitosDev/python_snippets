'''
NLP_to_run_SQL_queries (TAPEX).py
https://huggingface.co/microsoft/tapex-base

'''



from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base")



data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)


import carlos_utils.file_utils as fu
import EVC_utils.lesson_analysis_settings as lse
analysis_settings = lse.get_analysis_settings()
hc_root_folder = analysis_settings["hyperclass_root_folder"]
output_folder = fu.fullfile(hc_root_folder, 'aggregated_results')
df_students = fu.readPickleFile(fu.fullfile(output_folder, 'df_students.pickle'))


# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "select attendance_token where hyperclass_flag is not null"
encoding = tokenizer(table=df_students, query=query, return_tensors="pt")

query = "How many rows?"
encoding = tokenizer(table=df_students[0:100], query=query, return_tensors="pt")


outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['2008']