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


pip3 uninstall lightgbm

pip3 install lightgbm --install-option="--openmp-include-dir=/usr/local/opt/libomp/include/" --install-option="--openmp-library=/usr/local/opt/libomp/lib/libomp.dylib"


LightGBM
--------

import lightgbm as lgb

Variable PermutationImportanceprint('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()