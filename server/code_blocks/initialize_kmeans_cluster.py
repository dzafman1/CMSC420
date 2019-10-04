def initialize_kmeans_cluster(loaded_dataset, intermediate_df, description, method):

  # begin helper method
  def initializeClustersForKmeans(rawDf):
    '''Use k-means++ to initialize a good set of centroids'''
    from pandas.api.types import is_numeric_dtype
    from sklearn.metrics import pairwise_distances
    import numpy as np
    import pandas as pd
  
    df = rawDf.fillna(0) # replace any nulls with zero
    k = min(len(df),50) # there should not be more clusters than there are datapoints
    quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
    if len(quantitativeColumns) == 0 or len(df) == 0: # no quantitative columns/rows
      return pd.DataFrame([])
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
    # end helper method

  # begin block
  df= loaded_dataset
  res_df = initializeClustersForKmeans(df)

  res = {
    'output': df.head(10).to_json(orient='table'),
    'result': res_df.head(10).to_json(orient='table'),
    'description' : description,
    'type': method
  }
  intermediate_df.append(df.head(10))
  return res
  # end block

res = initialize_kmeans_cluster(self.current_df, self.intermediate_df, description, method)
