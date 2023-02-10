'''
visual_bert.py

>> It doesn't work
'''



Follow the tutorial from:
/Users/carlos.aguilar/Documents/EF_repos/transformers/examples/research_projects/visual_bert


# https://huggingface.co/transformers/model_doc/visual_bert.html
## Visual BERT
# VisualBERT is a multi-modal vision and language model.
#  It can be used for visual question answering, multiple choice, 
# visual reasoning and region-to-phrase correspondence tasks. 
# VisualBERT uses a BERT-like transformer to prepare embeddings for image-text pairs.
# Both the text and visual features are then projected to a latent space 
# with identical dimension.
# https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing#scrollTo=7-5rqN-vtlkq
import torch
from transformers import BertTokenizer, VisualBertModel

def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]

visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]


model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("What is the man eating?", return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path
visual_embeds = get_visual_embeddings(image_path)

visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask
})
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state


# Follow this
# https://github.com/huggingface/transformers/blob/master/examples/research_projects/lxmert/demo.ipynb
# code in here:
# /Users/carlos.aguilar/Documents/EF_repos/transformers/examples/research_projects/visual_bert


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
question1 = 'What is the girl standing on?'
question2 = 'What is the girl standing on?'
questions = [question1, question2]
tokens = tokenizer(questions, padding='max_length', max_length=50)

input_ids = torch.tensor(tokens["input_ids"])
attention_mask = torch.tensor(tokens["attention_mask"])
token_type_ids = torch.tensor(tokens["token_type_ids"])

visual_embeds = torch.stack(visual_embeds)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)

model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")


