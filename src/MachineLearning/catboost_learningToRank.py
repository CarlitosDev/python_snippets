'''

catboost_learningToRank.py

https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb


See https://www.wikiwand.com/en/Learning_to_rank
for a wider view


msrank_10k
----------

The dataset consists of feature vectors extracted from query-url pairs along with relevance judgment labels:

(1) The relevance judgments are obtained from a retired labeling set of a commercial web search engine (Microsoft Bing), 
which take 5 values from 0 (irrelevant) to 4 (perfectly relevant).

(2) The features are basically extracted by us, and are those widely used in the research community.


In this dataset queries and urls are represented by IDs. 



Each row corresponds to a query-url pair. The first column is relevance LABEL of the pair, 
the second column is query id, and the following columns are features. 
The larger value the relevance label has (0-4), the more relevant the query-url pair is. 
A query-url pair is represented by a 136-dimensional feature vector



The training dataset contains 10000 objects. Each object is described by 138 columns. 
The first column contains the label value, the second one contains the identifier of the object's group (GroupId). 
All other columns contain features.


'''


from collections import Counter
Counter(y_train).items()