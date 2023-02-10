Installation from source
------------------------

brew install cmake
brew install libomp
brew install gcc@8


'''
pip3 uninstall lightgbm
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
export CXX=g++-8 CC=gcc-8
mkdir build ; cd build
cmake ..
make -j4
cd ..
cd python-package; 
sudo python3 setup.py install
'''





# For Mojave (10.14)
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build


cmake \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib \
  ..

pip3 install --no-binary :all: lightgbm

# updates 16.02.2022. Why not doing this instead of the pip3 install above?
# cd ./../python-package/
# python3 setup.py install


pip3 uninstall lightgbm

pip3 install lightgbm --install-option="--openmp-include-dir=/usr/local/opt/libomp/include/" --install-option="--openmp-library=/usr/local/opt/libomp/lib/libomp.dylib"

pip3 install lightgbm --upgrade --user


LightGBM
--------

import lightgbm as lgb

Variable PermutationImportanceprint('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()






X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                  "B": np.random.permutation([1, 2, 3] * 100),  # int
                  "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                  "D": np.random.permutation([True, False] * 150)})  # bool
y = np.random.permutation([0, 1] * 150)
X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),
                       "B": np.random.permutation([1, 3] * 30),
                       "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                       "D": np.random.permutation([True, False] * 30)})
for col in ["A", "B", "C", "D"]:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': -1
}
lgb_train = lgb.Dataset(X, y)
gbm3 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False,
                 categorical_feature=['A', 'B', 'C', 'D'])





### We can also train (not for ranking) using the dataset functionality
categorical_features = inputVarTypes['input_cat_vars']

lgb_train = lgb.Dataset(X_train, label=y_train, 
  group=elements_in_query_train,
  categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_eval, label=y_eval, 
  reference=lgb_train, 
  group=elements_in_query_eval,
  categorical_feature=categorical_features)