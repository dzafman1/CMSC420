## Calculating the accuracy of the model
##def accuracy(predictions, labels):
#    # getting the percent of how many predictions were correct
#    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

# Calculating the accuracy of the model
def evaluateModelPredictions(df):
  import numpy as np
  import pandas as pd

  predictions = df.iloc[:,-1].values # last column
  labels = df.iloc[:,-2].values # second to last column
  return pd.DataFrame(np.equal(predictions,labels))
