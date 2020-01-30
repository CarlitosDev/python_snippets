Isolation forest distance-less regressors/classifiers available in sklearn
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.


# Categorical variable conversion
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# Invert the transform
class_le.inverse_transform(y)


# Count the number of words
from sklearn.feature_extraction.text import CountVectorizer
plainText = df['Campaign Name'].tolist()
vect = CountVectorizer(min_df=0., max_df=1.0)
X = vect.fit_transform(plainText)

