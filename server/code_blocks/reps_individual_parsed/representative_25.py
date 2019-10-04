#x = np.linspace(-1, 1, N)  # choose solution
#b = A.dot(x)               # sparse matrix vector product
#import scipy.sparse.linalg
#x = scipy.sparse.linalg.spsolve(A, b)
#print x

def solveSparseLinearSystem(df):
  import numpy as np
  import scipy.sparse.linalg
  from pandas.api.types import is_numeric_dtype
  from scipy import sparse
  import pandas as pd

  nCols = [c for c in list(df) if is_numeric_dtype(df[c])]
  A = df[nCols[:-1]].values
  if A.shape[0] != A.shape[1]:
    return None
  x = df[[nCols[-1]]].values

  b = A.dot(x)
  sA = sparse.csr_matrix(A)
  x = scipy.sparse.linalg.spsolve(sA, b)
  
  return pd.DataFrame(x)
