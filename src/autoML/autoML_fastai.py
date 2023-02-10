'''
autoML_fastai

https://www.kaggle.com/syzymon/covid19-tabnet-fast-ai-baseline
pip3 install fastai2
pip3 install fast_tabnet
'''

from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *
import pandas as pd


# Fake sales
num_samples = 500
num_features = 5
input_vars = [f'x_{idx}' for idx in range(1,num_features+1)]
input_data = np.random.rand(num_samples, num_features)

weights = np.array([13,9,6,1,0])
y_train = np.dot(input_data, weights.T)

df = pd.DataFrame(input_data, columns=input_vars)
df['response_var'] = y_train

splits = RandomSplitter()(range_of(df))

# Basic function to preprocess tabular data before assembling it in a `DataLoaders`.
to = TabularPandas(df, y_names='response_var', cont_names=input_vars, splits=splits)
dls = to.dataloaders()
learn = tabular_learner(dls, loss_func=[rmse])
learn.fit_one_cycle(1)
learn.show_results()







# or this option...
dls = TabularDataLoaders.from_df(df,y_names='response_var', \
    valid_idx=list(range(400,500)), cont_names=input_vars, splits=splits)
learn = tabular_learner(dls, loss_func=[rmse])





'''
# Another example
dls = TabularDataLoaders.from_df(df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                 y_names="salary", valid_idx=list(range(800,1000)), bs=64)
'''

'''
procs=[FillMissing, Categorify, Normalize]

splits = list(range(MAX_TRAIN_IDX)), (list(range(MAX_TRAIN_IDX, len(df))))

to = TabularPandas(df1, procs, cat_vars.copy(), cont_vars.copy(), dep_var, y_block=TransformBlock(), splits=splits)
'''

dls = to.dataloaders(bs=512, path=path)
dls.show_batch()

emb_szs = get_emb_sz(to); print(emb_szs)


# Model
dls.c = 2 # Number of outputs we expect from our network - in this case 2.
model = TabNetModel(emb_szs, len(to.cont_names), dls.c, n_d=32, n_a=32, n_steps=3)
opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dls, model, MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[rmse])
learn.lr_find()