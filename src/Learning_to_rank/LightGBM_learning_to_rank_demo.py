'''
Learning to rank With LightGBM

https://tamaracucumides.medium.com/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574

LambdaRank as objective function. LambdaRank has proved to be very effective on optimizing ranking functions such as nDCG. If you want to know more about LambdaRank, go to this article: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/

Normalized discounted cummulative gain (nDCG)

'''

import lightgbm as lgb
gbm = lgb.LGBMRanker()



from catboost.datasets import msrank_10k
train_df, test_df = msrank_10k()

# each document (user query?) has got 136 features
# column 0 is the label (from 0 to 4)
# column 1 is the query (87 different queries)
# >> basically what I see here is that there are 87 different 
# queries that have returned documents with 136 features and a relevance from 0 to 4.
train_df.head()


X_train = train_df.drop([0, 1], axis=1)
y_train = train_df[0]
queries_train = train_df[1]

X_test = test_df.drop([0, 1], axis=1)
y_test = test_df[0]
queries_test = test_df[1]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


query_train = [X_train.shape[0]]
query_val = [X_val.shape[0]]
query_test = [X_test.shape[0]]

gbm.fit(X_train, y_train, group=query_train,
        eval_set=[(X_val, y_val)], eval_group=[query_val],
        eval_at=[5, 10, 20], early_stopping_rounds=50)


test_pred = gbm.predict(X_test)

X_test["predicted_ranking"] = test_pred
X_test.sort_values("predicted_ranking", ascending=False)
