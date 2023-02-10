cd '/Users/carlos.aguilar/Documents/repos_and_software'
git clone https://github.com/pytorch/vision.git
cd vision
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python3 setup.py install
cd ..
python3 -c 'import torchvision as tv; print(tv.__version__)'