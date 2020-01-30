from imblearn.under_sampling import RandomUnderSampler
X = df.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis = 1)
y = df.isFraud
rus = RandomUnderSampler(sampling_strategy=0.8)
X_res, y_res = rus.fit_resample(X, y)
print(X_res.shape, y_res.shape)
print(pd.value_counts(y_res))