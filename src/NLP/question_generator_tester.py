question_generator_tester.py


'''

cd /Users/carlos.aguilar/Documents/EF_repos
git clone https://github.com/amontgomerie/question_generator/

cd question_generator/


Quick test:
python3 'run_qg.py' --text_dir './articles/twitter_hack.txt'



'''

'''
# if not using the wrapper class
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
# model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model.to(device)
'''



from questiongenerator import QuestionGenerator
from questiongenerator import print_qa

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

qg = QuestionGenerator()


# /Users/carlos.aguilar/Documents/EF_Content/EFxAWS/lumiere_data/GeneralEnglish_analysis_results/json/GE_11.5.3.2.1_analysis.json
this_transcription = '''creativity is important not only for artists and writers, 
but also for people who work in professions such as business architecture, science engineering. 
The world we live in today is driven by innovation and in order to have innovation, you need creativity. 
Creativity is not something that only a lucky few are born with. In fact it can be developed. 
Here are some ways you can inspire refresh and enhance your creativity routines. 
Send your brain into a comfort zone when you do the same things every day, go to work the same way, 
eat the same food, read the same newspapers, your brain goes on autopilot and stops 
being stimulated by fresh perspectives, try to keep things fresh, Do things differently, 
take a different route to work. The freshness will stimulate your mind, keeping it active, 
leading to more creativity. Creativity happens when the brain is stimulated.
A new environment is the most stimulating environment. Think of babies. Everything around them is new. 
They are filled with curiosity and they love to explore everything. See the area around you as 
a tourist would see it, you might see something new and it might spark a new idea. 
Mind maps are used to generate and visualize ideas. They are a brainstorming technique 
which helps to show connections between concepts often leading to new creative ideas. 
Write a central concept on your paper, add branches, writing words which are related 
to the central concept, use lines, colors, arrows, branches or some other way of showing 
connections between the ideas generated on your mind map right quickly, trying not to limit yourself. 
Leave lots of space. You can come back later and add more ideas. You'll be surprised at how a mind
 map can move you towards that great idea. What if questions can take you on a creative conceptual journey? 
 Who knows what crazy ideas you might come up with? What if 12 year olds could drive CASS? What if 
 skateboards could fly? What if toasters could make cheeseburgers? What if strengthening your creative 
 muscles is something that can help you in all aspects of your life? Try these things and see if you can get 
 those creative juices flowing, you might have the next great idea that changes the world.'''

qa_list = qg.generate(
    this_transcription, 
    num_questions=4, 
    answer_style='all'
)
print_qa(qa_list)


'''

1) Q: What is the best way to stimulate your mind?
   A: The freshness will stimulate your mind, keeping it active,</s> leading to more creativity. 

2) Q: What can you do to stimulate your mind to think differently?
   A: </s> Send your brain into a comfort zone when you do the same things every day, go to work the same way,</s> eat the same food, read the same newspapers, your brain goes on autopilot and stops</s> being stimulated by fresh perspectives, try to keep things fresh, Do things differently,</s> take a different route to work. 

3) Q: What can you do to stimulate your mind?
   A: </s> Here are some ways you can inspire refresh and enhance your creativity routines. 

4) Q: What if the mind could be a tool to stimulate your creativity?
   A: What if 12 year olds could drive CASS? 

'''


qa_list_MC = qg.generate(
    this_transcription, 
    num_questions=4, 
    answer_style='multiple_choice'
)
print_qa(qa_list_MC)

'''
1) Q: What is the world we live in today?
   A: 1. 12 year olds 
      2. today (correct)

2) Q: What if 12 year olds could drive CASS?
   A: 1. 12 year olds (correct)
      2. today 
'''



####### Compare to Quillionz

'''

----Factual Questions----

1) What if _______ can take you on a creative conceptual journey?   
Answer: questions 

2) You can come back later and add more _______.   
Answer: ideas 

3) _______ _______ are used to generate and visualize ideas.   
Answer: Mind maps 

4) What if toasters could make cheeseburgers? What if strengthening your _______ _______ is something that can help you in all aspects of your life?   
Answer: creative muscles 

5) Who knows what _______ _______ you might come up with?   
Answer: crazy ideas 

6) They are a brainstorming technique which helps to show connections between concepts often leading to new creative _______.   
Answer: ideas 

7) The freshness will stimulate your _______, keeping it active, leading to more creativity.   
Answer: mind 

8) What if 12 year olds could drive _______?   
Answer: CASS 

9) You can come back later and add more ideas. You'll be surprised at how a mind map can move you towards that _______ _______.   
Answer: great idea 

10) The _______ we live in today is driven by innovation and in order to have innovation, you need creativity.   
Answer: world 

11) Here are some ways you can inspire refresh and enhance your _______ _______.   
Answer: creativity routines 

12) Try the things and see if you can get those creative juices flowing, you might have the next _______ _______ that changes the world.   
Answer: great idea 

13) Describe 'New environment'.    
Answer source: A new environment is the most stimulating environment. 


----Premium Questions----

14) What is a central concept on the paper?     
Answer: Add branches, writing words. 

15) What is used to generate and visualize ideas?     
Answer: Mind maps. 

16) What is the most sensitive environment?     
Answer: New environment. 


----Interpretive Questions----

17) What is creativity?   
Answer: important not only for artists and writers, but also for people who work in professions 

18) What kind of professions does it help?    
Answer: business architecture, science engineering 

19) When do we live in the present day?    
Answer:  

20) When does creativity happen?    
Answer: when the brain is stimulated 


---------Notes----------

     Creativity is not something that only a lucky few are born with, it can be developed. It is important not only for artists and writers, but also for people who work in professions such as business architecture, science engineering. Here are some ways you can inspire refresh and enhance your creativity routines. Try these things and see if you can get those creative juices flowing, you might have the next great idea that changes the world. Do things differently, take a different route to work, or a new environment is the most stimulating environment. A mind map is a brainstorming technique which helps to show connections between concepts often leading to new creative ideas.



'Created by Quillionz | World's first AI-powered platform for building quizzes and notes'






￼
￼





Another example - Unilever

(Clean Future How we’re turning off the tap on fossil fuels_resized_analysis.json)

"bert_extractive": "It means hygiene, self worse and dignity. We are using the power of nature bad with science to create a completely new bio surfactants and to unlock the potential off Great carbon were breaking down known recyclable plastic waste and using the carbon to create biodegradable ingredients. There is our own part ofthe gold created by renewable and recycle carbon. The common rainbow is all about keeping and using the carbon in the loop.",


"transcription": "all around world cleaning has meaning. It means hygiene, self worse and dignity. Some people, including myself, even find a sense of Zen in cleaning. But can we find a way to clean the world by cleaning our homes? The global cleaning industry has twice the water consumption off the EU, which is pollutes with non biodegradable chemicals and admits as much carbon as my country. The nettle. It's cleaning products I largely made from fossil fuels sold in plastic bottles and leave persistent ingredients in nature. This might sound like a crazy idea. With what if we could clean up some of the world's carbon, plastic and water pollution whilst cleaning. Our hope's Unilever has committed to net zero by 2039 a decade earlier than the Paris Agreement. To meet this ambitious target, we must address a hidden culprit in our products. Fossil derived chemicals most off the force or common chemicals will find its way into the apathy. So the more we extract from underground Maur, we adding to the problem buster ground. Our challenges don't to stop using calm. How could be its instead to use the right sources off com sources that don't end most. You're to tow the atmosphere, using different sources off carbon for our products. You're probably thinking, can this be done or is this science fiction? The answer is it can you could call it a carbon revolution. We call it the carbon rainbow. Now, let me explain. This black carbon is what industries including ours are addicted to coal, oil and gas. We shall pumped from under the ground to make our cleaning products, we have decided to progressively turn off the tap and are replacing black carbon with this. This is two carbon rainbow which shows the variety of carbon sources from above the ground to produce purple carbon. We are capturing carbon emissions from industrial processes and transforming them into cleaning ingredients like soda ash for washing powders to create blue carbon. We are exploring oceans to find more sustainable solutions like enzymes that clean and cold water to make green carbon. We are using the power of nature bad with science to create a completely new bio surfactants and to unlock the potential off Great carbon were breaking down known recyclable plastic waste and using the carbon to create biodegradable ingredients. Now, at the end of the rainbow. There is our own part ofthe gold created by renewable and recycle carbon. We take these new ingredients and reformulate our products, aiming to keep the cost affordable to consumers and was to save, if not better, cleaning performance. They are used and thrown away and look what happens here. We chemically recycle plastic waste, and it goes into the cycle as biodegradable grey carpet, helping to solve some off the plastic problem and a purple, blue and green carbon biodegrade. Every turn to the cycle, too. No black carbon and yet great cleaning. The common rainbow is all about keeping and using the carbon in the loop. It's better for the planet. It's equally good for the business consumers. Laffitte and these technologies give us long lasting, competitive Etch. For the first time since industrial pollution, we have the opportunity to choose which style of car be use by simply tapping into the complete geo comes is loss of planet. Do you still think this is science fiction? Letters explain how it is happening already to fucky. Cleaning ingredients are surfactants and soda ash. They play an important role in cleaning products to make them form, remove dirt and perform at the highest level, but they had made using a lot of black carbon. Inspired by nature, we're working, but by technology companies such as Ivanic to create new generations of surfactant called Ram lipids. Super effective, biodegradable, renewable and has reduced the carbon footprint offer formula by a massive 28%. And in partnership with Utica and helpfully chemicals and fertilisers, I'm carbon clean, we capturing carbon emissions and turning it into soda ash. It's almost magic, but who really cares? How renewable are communists? If we demonstrate environment in the process as we move forward, replacing fossil carbon with the carbon rainbow must be done responsibly in our transition away from virgin fossil carbon were committed to protecting nature. Ending deforestation will always take precedence. We will develop new tools, innovations, techniques and programmes on the ground that help us deliver impacts at the level of field of farm and landscape. And we will do this in a way that is inclusive of communities and that empowers farmers and smallholders. Fossil carbon in chemicals must no longer be the hidden culprit off climate change We need to tackle the issue handled. If we're going to reach our net zero objective. If cities like New York and Copenhagen already being built, was reused common concrete, then it is about time that ball consumer goods from Elektronik do closing to cleaning products embrace it too. The beauty ofthe the rainbow ist that it is leveraging existing industrial infrastructure and the billions off investments in the system we already have. But to create the future, we need to collaborate across sectors we need to do it together. We collectively need a road map out off petrochemicals once and for all. The question is, will you be part ofthe the movement?"


 'Quillionz | Questions and Notes in seconds' 

Table of Contents
Factual Questions...............................................
Premium Questions...............................................
Interpretive Questions..........................................
Notes.........................................................


----Factual Questions----

1) We are capturing carbon emissions from industrial processes and transforming them into cleaning ingredients like _______ _______ for washing powders to create blue carbon.   
Answer: soda ash 

2) And in partnership with _______ and helpfully chemicals and fertilisers, I'm carbon clean, we capturing carbon emissions and turning it into soda ash.   
Answer: Utica 

3) Our challenges don't to stop using calm. How could be its instead to use the _______ _______ off com sources that don't end most.   
Answer: right sources 

4) And in partnership with Tampa and helpfully chemicals and fertilisers, I'm carbon clean, we capturing carbon emissions and turning it into soda ash.   
Answer: False 
Correct Sentence: And in partnership with Utica and helpfully chemicals and fertilisers, I'm carbon clean, we capturing carbon emissions and turning it into soda ash. 

5) Now, let me explain. This _______ _______ is what industries including ours are addicted to coal, oil and gas.   
Answer: black carbon 

6) So the more we extract from underground _______, we adding to the problem buster ground.   
Answer: Maur 

7) Explain the following with an example: 'Technology companies'    
Answer source: Inspired by nature, we're working, but by technology companies such as Ivanic to create new generations of surfactant called Ram lipids. 

8) Our hope's Unilever has committed to net zero by _______ a decade earlier than the Paris Agreement.   
Answer: 2039       
A)2039       
B)2029       
C)2023       
D)2032   

9) Our hope's _______ has committed to net zero by 2039 a decade earlier than the Paris Agreement.   
Answer: Unilever 

10) Who give us long lasting, competitive Etch?   
Answer: Laffitte and these technologies 

11) _______ _______ in chemicals must no longer be the hidden culprit off climate change We need to tackle the issue handled.   
Answer: Fossil carbon 

12) The answer is it can you could call it a _______ _______.   
Answer: carbon revolution 

13) If we're going to reach our _______ _______ objective.   
Answer: net zero 

14) There is our own part ofthe gold created by renewable and _______ _______.   
Answer: recycle carbon 

15) It's cleaning products I largely made from fossil fuels sold in _______ _______ and leave persistent ingredients in nature.   
Answer: plastic bottles 

16) The _______ _______ is all about keeping and using the carbon in the loop.   
Answer: common rainbow 


----Premium Questions----

17) Who inspired the new generations of surfactant?     
Answer: Ram lipids. 

18) What is the water consumption off the eu?     
Answer: Pollutes with non biodegradable chemicals. 

19) What was expected to reach the net zero objective?     
Answer: 're going. 


----Interpretive Questions----

20) What is the hidden problem   
Answer: culprit 

21) What will happen to the environment   
Answer: We shall pumped from under the ground to make our cleaning products, we have decided to progressively turn off the tap and are replacing black carbon with this. 

22) What is the rainbow?   
Answer: carbon 

23) What are they addicted to?   
Answer: coal, oil and gas 


---------Notes----------

     The global cleaning industry has twice the water consumption off the EU, pollutes with non biodegradable chemicals and admits as much carbon as my country. Cleaning products I largely made from fossil fuels sold in plastic bottles and leave persistent ingredients in nature. Unilever has committed to net zero by 2039 a decade earlier than the Paris Agreement. We are capturing carbon emissions from industrial processes and transforming them into cleaning ingredients like soda ash for washing powders to create blue carbon. We are exploring oceans to find more sustainable solutions like enzymes that clean and cold water to make green carbon. At the end of the rainbow there is our own part ofthe gold created by renewable and recycle carbon. The common rainbow is all about keeping and using the carbon in the loop. Cleaning ingredients are surfactants and soda ash, but they had made using a lot of black carbon. Fossil carbon in chemicals must no longer be the hidden culprit off climate change. The beauty ofthe the rainbow ist that it is leveraging existing industrial infrastructure and the billions off investments in the system we already have. We collectively need a road map out off petrochemicals once and for all. The question is, will you be part of the movement? We need to collaborate across sectors we need to do it together. But to create the future of the future, we need collaboration across sectors.



'Created by Quillionz | World's first AI-powered platform for building quizzes and notes'

'''