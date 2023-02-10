python_surprise.py


import pandas as pd
import surprise



# This library needs the datasets and training sets in their own format.

# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is requiered.
reader = surprise.Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = surprise.Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
algo = surprise.SVDpp()

pkf = surprise.model_selection.PredefinedKFold()

for trainset, testset in pkf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    surprise.accuracy.rmse(predictions, verbose=True)
