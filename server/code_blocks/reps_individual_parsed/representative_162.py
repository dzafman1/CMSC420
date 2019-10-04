#def smart_initialize(data, k, seed=None):
#    '''Use k-means++ to initialize a good set of centroids'''
#    if seed is not None: # useful for obtaining consistent results
#        np.random.seed(seed)
#    centroids = np.zeros((k, data.shape[1]))
#    
#    # Randomly choose the first centroid.
#    # Since we have no prior knowledge, choose uniformly at random
#    idx = np.random.randint(data.shape[0])
#    centroids[0] = data[idx,:].toarray()
#    # Compute distances from the first centroid chosen to all the other data points
#    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
#    
#    for i in xrange(1, k):
#        # Choose the next centroid randomly, so that the probability for each data point to be chosen
#        # is directly proportional to its squared distance from the nearest centroid.
#        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
#        idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
#        centroids[i] = data[idx,:].toarray()
#        # Now compute distances from the centroids to all data points
#        squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
#    
#    return centroids


def initializeClustersForKmeans(df):
  '''Use k-means++ to initialize a good set of centroids'''
  import numpy as np
  import pandas as pd
  from pandas.api.types import is_numeric_dtype
  from sklearn.metrics import pairwise_distances

  k = 50
  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  centroids = np.zeros((k, len(quantitativeColumns)))
  data = df[quantitativeColumns].values

  # Randomly choose the first centroid.
  # Since we have no prior knowledge, choose uniformly at random
  idx = np.random.randint(data.shape[0])
  centroids[0] = data[idx,:]
  # Compute distances from the first centroid chosen to all the other data points
  squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
    
  for i in xrange(1, k):
    # Choose the next centroid randomly, so that the probability for each data point to be chosen
    # is directly proportional to its squared distance from the nearest centroid.
    # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
    idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
    centroids[i] = data[idx,:]
    # Now compute distances from the centroids to all data points
    squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
    
  final = {}
  for i,c in enumerate(quantitativeColumns):
    final[c] = centroids[:,i]
  return pd.DataFrame(final)
