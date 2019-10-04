#import numpy as np
#
## Summarize the data about minutes spent in the classroom
#total_minutes = total_minutes_by_account.values()
#print 'Mean:', np.mean(total_minutes)
#print 'Standard deviation:', np.std(total_minutes)
#print 'Minimum:', np.min(total_minutes)
#print 'Maximum:', np.max(total_minutes)

# Leilani's example from the paper
def getStats(df): 
  import numpy as np
  import pandas as pd
  from pandas.api.types import is_numeric_dtype
  
  nCols = [c for c in list(df) if is_numeric_dtype(df[c])]
  data = {
    "numericColumns": nCols, "mean":[],
    "std":[], "min":[], "max":[]
  }
  for c in nCols:
    d = df[[c]]
    data["mean"].append(np.mean(d.values))
    data["std"].append(np.std(d.values))
    data["min"].append(np.min(d.values))
    data["max"].append(np.max(d.values))
  return pd.DataFrame(data)
