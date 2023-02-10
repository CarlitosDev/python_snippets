from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)


conversation = Conversation()




text = 'I want to read a new book'
conversation.add_user_input(text)

result = nlp([conversation], do_sample=False, max_length=1000)

messages = []
for is_user, text in result.iter_texts():
    current_message = {
        'is_user': is_user,
        'text': text
    }
    messages.append(current_message)
    print(f'{current_message}')