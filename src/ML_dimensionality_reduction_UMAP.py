import pandas as pd
import numpy as np 
# pip3 install umap-learn
import umap.umap_ as umap

df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 80)))
print(df.shape)

# Reduce to 2 dimensions
reducer = umap.UMAP()
embedding = reducer.fit_transform(df.values)
embedding.shape