'''

	xgboost only deals with numeric columns.

	installation June 2019

	brew install gcc@8
	gcc-8 --version
	brew switch gcc 8.3.0
	brew link --overwrite gcc


	brew uninstall gcc@9


pip3 uninstall xgboost


git clone --recursive https://github.com/dmlc/xgboost

cd xgboost
export CXX=g++-8 CC=gcc-8
mkdir build ; cd build
cmake ..
make -j4
cd ..
cd python-package; 
sudo python3 setup.py install



??CC=gcc-8 CXX=g++-8 cmake .. -DR_LIB=ON
'''


'''
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
CC=gcc-8 CXX=g++-8 cmake ..
make -j4

Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include/sys/resource.h:447:34: error: expected initializer before '__OSX_AVAILABLE_STARTING'
'''


Use LIME with XGBoost
http://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html




Incremental training:


params.update({'process_type': 'update',
               'updater'     : 'refresh',
               'refresh_leaf': True})
model_2_v2_update = xgb.train(params, xg_train_2, 30, xgb_model=model_1)







# Regular XGBoost
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df[inputVars], df[responsevar_bin], test_size=0.2)


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
# specify parameters via map, definition are same as c++ version
param = {'n_estimators': 100, 'max_depth':2, 'eta':0.1, 'silent':0, 'objective':'binary:logistic'}

# specify validations set to watch performance
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dvalid)
labels = dvalid.get_label()





'''

Feature importance

# https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7

- Weight: is the percentage representing the relative number of 
times a particular feature occurs in the trees of the model. 

- Gain implies the relative contribution of the corresponding feature to the model 
calculated by taking each feature’s contribution for each tree in the model. 
A higher value of this metric when compared to another feature implies it 
is more important for generating a prediction.

- Coverage:relative number of observations related to this feature. It measures 
the relative quantity of observations concerned by a feature.

'''

# importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
importance_type = 'gain'
col_names = [importance_type + '_' + 'totals']
d_t = model.get_booster().get_score(importance_type=importance_type)
df_features = pd.DataFrame.from_dict(d_t,  orient='index', columns = col_names)
df_features.sort_values(by=col_names, ascending=False, inplace=True)
df_features.reset_index(inplace=True)
# to plot it:
xgb.plot_importance(model)
plt.show()
