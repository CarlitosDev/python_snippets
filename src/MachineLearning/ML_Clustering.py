ML_Clustering.py


from sklearn.cluster import DBSCAN, OPTICS, MiniBatchKMeans


if do_DBSCAN:
    clustering1 = DBSCAN(eps=1, min_samples=5).fit(input_non_zero)
elif do_OPTICS:
    clustering1 = OPTICS(min_samples=5).fit(input_non_zero)
else:
    clustering1 = MiniBatchKMeans(n_clusters=3).fit(input_non_zero)


labels = clustering1.labels_


# Classic k-means++

# Quick trick below to ignore the values that are zero 
# ignoring 0 sales in this classification
idx_zero_sales = input_signal==0
input_non_zero = input_signal[~idx_zero_sales].reshape(-1,1)

sales_mu = input_non_zero.mean()

clusters = MiniBatchKMeans(n_clusters=3).fit(input_non_zero)
centroids = clusters.cluster_centers_.squeeze()
idxSorted = np.argsort(centroids)
label_map = {idxSorted[idx]:idx for idx in range(0, len(idxSorted))}

mini_labels = np.vectorize(label_map.get)(clusters.labels_) - clusters.labels_.min() + 1