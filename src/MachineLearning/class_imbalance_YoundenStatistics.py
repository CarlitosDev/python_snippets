



# For example, if we have the output from CatBoost
y_hat_prob = model.predict_proba(df_eval[inputVars_modelling])
prob_class_0 = y_hat_prob[:, 0]
prob_class_1 = y_hat_prob[:, 1]



from sklearn.metrics import classification_report, roc_curve
# Use Youden’s J statistic to decide the _optimal_ threshold
#
# Taken from https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
# keep probabilities for the positive outcome only
# calculate roc curves
fpr, tpr, thresholds = roc_curve(df_eval[responseVar], prob_class_1)
# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f{best_thresh}')

# 
y_pred_Youdens = np.zeros(shape=y_pred.shape, dtype='bool')
idx_Youdens = prob_class_1 >= best_thresh
y_pred_Youdens[idx_Youdens] = True
print(classification_report(df_eval[responseVar], y_pred_Youdens))



'''
	Example with mixed probabilities that might lead to FP and FN
'''
y_eval = [0,0, 1,1,1,0]
prob_class_1 = np.array([0.3, 0.25,0.7,0.35,0.28,0.6])
fpr, tpr, thresholds = roc_curve(y_eval, prob_class_1, pos_label=1)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
best_thresh