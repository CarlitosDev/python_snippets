'''
sentdex_homemade_chatGPTR.py
'''

# Sentdex
# From https://github.com/Sentdex/ChatGPT-at-Home

import transformers
from transformers import utils, pipeline, set_seed
import torch
from flask import Flask, request, render_template, session, redirect



# Set the secret key for the session
MODEL_NAME = "facebook/opt-125m" 

# Initialize the chat history
history = ["Human: Can you tell me the weather forecast for tomorrow?\nBot: Try checking a weather app like a normal person.\nHuman: Can you help me find a good restaurant in the area\nBot: Try asking someone with a functioning sense of taste.\n"]
# generator = pipeline('text-generation', model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.half)
generator = pipeline('text-generation', model=f"{MODEL_NAME}", do_sample=True)



# Define the chatbot logic
def chatbot_response(input_text, history):
    # Concatenate the input text and history list
    input_text = "\n".join(history) + "\nHuman: " + input_text + " Bot: "
    set_seed(32)
    response_text = generator(input_text, max_length=1024, num_beams=1, num_return_sequences=1)[0]['generated_text']
    # Extract the bot's response from the generated text
    response_text = response_text.split("Bot:")[-1]
    # Cut off any "Human:" or "human:" parts from the response
    response_text = response_text.split("Human:")[0]
    response_text = response_text.split("human:")[0]
    return response_text


print(history)
input_text = 'It seems that is raining now'
response_text = chatbot_response(input_text, history)
print(response_text)

num_turns = 4
for idx in range(0, num_turns):

    input_text = input()
    response_text = chatbot_response(input_text, history)
    print(response_text)
    # Append the input and response to the chat history
    history.append(f"Human: {input_text}")
    history.append(f"Bot: {response_text}")