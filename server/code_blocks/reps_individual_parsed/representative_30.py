#from sklearn.ensemble import RandomForestClassifier
#
#forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)
#print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))

# Leilani's example from the paper
def testRandomForestClassifier(df):
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score
  from pandas.api.types import is_numeric_dtype
  import pandas as pd

  forest = RandomForestClassifier(n_estimators=100,
    n_jobs=-1,random_state=17)
  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  X_train = df[quantitativeColumns[:-1]]
  y_train = df[[quantitativeColumns[-1]]].values.ravel()
  return pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5))
