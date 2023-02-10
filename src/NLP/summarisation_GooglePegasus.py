'''

  I have tried this on the 02.03.2022. It looks great for summarising emails and articles.
  

'''


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))




a = '''
Hi carlos,

Would you like to be part of Monzo's growing Data Science team?

Someone with your level of experience would have a huge impact on the business and the direction our our products.

We recently closed off another round of funding, so there are some very exciting things on the horizon.

If you would like to have a chat to hear more let me know and we can get something arranged.

Mr Neil B
Technical Recruiter - Data Science at Monzo Bank
''' 
email_to_summarise = ' '.join([this_line for this_line in a.split('\n') if this_line != ''])

inputs = tokenizer(email_to_summarise, max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"])
this_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(this_summary)



transcription="Okay. In the afternoon, I went to Pete's. We had lunch and watched the match. Yeah. Thomas Good. Yeah. All right, man off. Yeah, Ladies do. Hey, are you going to throw that away? For what? Yeah. Can I have you take it? It would be great. I think it was my apartment. Take it. Thanks a lot. Later, Peter gave me a table at three. I started off for my job interview. My shoe broke. Yeah, I got to the station at 3. 30. I got to the station at 3. 30."
inputs = tokenizer(transcription, max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"])
this_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(this_summary)