From https://eng.uber.com/argos-real-time-alerts/

- outlier detector: compares whether in-streaming data is an inlier or outlier against pre-calculated thresholds
 and an outage detector. The small remainder of metrics classified as outliers are subsequently analyzed by the outage detector.
 -outage detector. The outage detector is computationally more complex and thus cannot be applied to all metrics with high frequency. Multivariate non-linear approach.


 Let the past be the past. 
 Past outages and outliers must not affect the outlier score for in-streaming data points. For example, during a thunderstorm in Chicago in June 2015 (see above image), ride demand increased 300%. That next week, we didn’t want our outlier algorithm to adapt to it. We utilize a median approach to allow for statistical robustness.



 https://colab.research.google.com/github/CarlitosDev/contrastiveExplanation/blob/master/contrastiveRegressor/Experiment2_surrogate_sales.ipynb

 https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk





from google.colab import drive
drive.mount('/content/drive')
project_path = '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations'
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/contrastiveRegressor.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/frc_runner.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/fcn_helpers.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/generate_Gompertz_sales.py' .
!pip install pandas catboost numpy matplotlib sklearn xlsxwriter ngboost category_encoders


from google.colab import drive
drive.mount('/content/drive')
project_path = '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations'
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/contrastiveRegressor.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/frc_runner.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/fcn_helpers.py' .
!cp '/content/drive/My Drive/order/Machine Learning Part/contrastive explanations/generate_Gompertz_sales.py' .
!pip install pandas catboost numpy matplotlib sklearn xlsxwriter ngboost category_encoders