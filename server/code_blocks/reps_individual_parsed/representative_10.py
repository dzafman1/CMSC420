#import numpy.linalg as LA
#LA.norm(u)

def calcNorm(df):
  import numpy.linalg as LA
  from pandas.api.types import is_numeric_dtype
  import pandas as pd
  nCols = [c for c in list(df) if is_numeric_dtype(df[c])]
  data = {"columnName":[],"NumpyLinalgNorm":[]}
  for nc in nCols:
    data["columnName"].append(nc)
    data["NumpyLinalgNorm"].append(LA.norm(df[[nc]].values))
		
  return pd.DataFrame(data)

