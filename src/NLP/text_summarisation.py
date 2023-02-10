'''
text_summarisation.txt
'''


abstract_cannibalisation = '''In food and grocery retail, sales cannibalisation during promotions occurs when a promoted product has a knock-on effect on the sales of a non-promoted one. The quantification of its effect is important for retailers, as cannibalisation can lead to wasted food and lost profits. The performance of promotions is ultimately dependent on their features but also on the characteristics of the stores, i.e. type of store or its location. Generally speaking, there is no homogeneous response to a promotion, and by extension to sales cannibalisation. Accordingly, in this paper we describe a framework to analyse the effects of cannibalisation due to individual promotions based on the relationship amongst their sales. The novelty of our work resides in understanding cannibalisation as a causal effect where the increase in sales of the promoted product is partly due to the decrease in sales of the non-promoted one. As such, we propose to use causal inference to measure the impact of cannibalisation due to promotions. Our method reviews each product that has been on promotion, searching for potential cannibals, given by products whose promotion have resulted in large sales uplifts, and for the fall-outs, given by those products experiencing a reduction in sales due to the cannibals. Then each cannibal-victim pair is analysed with Causal Impact, a time-series method which allows one to infer the causal effect of an intervention. We demonstrate the practical application of detecting cannibalisation on vast datasets of promotions within several stores and their many departments. To provide with an overview of the cannibalisation for entire departments and not simply for individual products, we build a directed graph. This is both unique and of utmost value to store managers and marketing teams. Additionally, we discuss in the Appendix the application of explainable forecasting to cannibalisation on a surrogate model.'''


# Which can be installed through pip3 install bert-extractive-summarizer
from summarizer import Summarizer
model = Summarizer()
bert_summary = model(abstract_cannibalisation, num_sentences=3)  # Will return 3 sentences 

# it returns
'''
'In food and grocery retail, sales cannibalisation during promotions occurs when a promoted product has a knock-on effect on the sales of a non-promoted one. Accordingly, in this paper we describe a framework to analyse the effects of cannibalisation due to individual promotions based on the relationship amongst their sales. As such, we propose to use causal inference to measure the impact of cannibalisation due to promotions. This is both unique and of utmost value to store managers and marketing teams.''''