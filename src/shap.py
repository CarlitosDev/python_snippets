SHapley Additive exPlantions (SHAP)[1]. It is introduced by Lundberg et al. 


https://towardsdatascience.com/interpreting-your-deep-learning-model-by-shap-e69be2b47893

Shapley value which is a solution concept in cooperative game theory.


The idea is using game theory to interpret target model. 
All features are “contributor” and trying to predict the task which is 
“game” and the “reward” is actual prediction minus the result from explanation model.


SHAP provides multiple explainers for different kind of models:

TreeExplainer: Support XGBoost, LightGBM, CatBoost and scikit-learn models by Tree SHAP.
DeepExplainer (DEEP SHAP): Support TensorFlow and Keras models by using DeepLIFT and Shapley values.
GradientExplainer: Support TensorFlow and Keras models.
KernelExplainer (Kernel SHAP): Applying to any models by using LIME and Shapley values.