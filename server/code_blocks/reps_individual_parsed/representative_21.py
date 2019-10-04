#all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
#all_cities

def outerJoin(df1,df2=None):
  import pandas as pd

  if df2 is None: # just return the original data frame
    return df1
  # find all common attributes across the data frames
  def findJoinKey(df1,df2): 
    toSearch = list(df2)
    for c1 in list(df1):
      if toSearch.index(c1) >= 0:
         return c1
    return None
  joinKey = findJoinKey(df1,df2)
  if joinKey is None:
    return None
  return pd.merge(left=df1, right=df2, on=joinKey, how="outer")
