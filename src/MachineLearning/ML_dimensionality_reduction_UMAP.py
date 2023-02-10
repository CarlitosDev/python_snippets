'''
Uniform Manifold Approximation and Projection (UMAP) 
is a dimension reduction technique that can be used for visualisation similarly to t-SNE, 
but also for general non-linear dimension reduction. 

The algorithm is founded on three assumptions about the data

- The data is uniformly distributed on Riemannian manifold;
- The Riemannian metric is locally constant (or can be approximated as such);
- The manifold is locally connected.

https://arxiv.org/abs/1802.03426

https://umap-learn.readthedocs.io/en/latest/

'''

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

#https://umap-learn.readthedocs.io/en/latest/parameters.html



# More tests on 
#.../Google Drive/order/Machine Learning Part/Preparing the 3rd paper/store_sales_projection.ipynb


'''

Updates 21.10.2020

UMAP supervised (https://umap-learn.readthedocs.io/en/latest/supervised.html)
You can train a UMAP projection and embed new points. So, can I train UMAP on cannibalisation scenarios and use the learnings? Making use of the labels

It has got .fit and .transform methods in the likes of sklearn.


Combine spaces with different metrics to include categorical data. Haversine, Levenstein (sparse)


Good talk by the creator
https://www.youtube.com/watch?v=nq6iPZVUxZU


Some Google guys explaining it:

https://pair-code.github.io/understanding-umap/
with these great remarks: 
"UMAP is an incredibly powerful tool in the data scientist's arsenal, and offers a number of advantages over t-SNE.
While both UMAP and t-SNE produce somewhat similar output, the increased speed, better preservation of global structure, 
and more understandable parameters make UMAP a more effective tool for visualizing high dimensional data. 
Finally, it's important to remember that no dimensionality reduction technique is perfect - by necessity, 
we're distorting the data to fit it into lower dimensions - and UMAP is no exception. However, by building up an 
intuitive understanding of how the algorithm works and understanding how to tune its parameters, we can more 
effectively use this powerful tool to visualize and understand large, high-dimensional datasets."



'''



# Supervised UMAP
embedding = umap.UMAP().fit_transform(data, y=target)




def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric)
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], s=80)
    plt.title(title, fontsize=18)
    return u
