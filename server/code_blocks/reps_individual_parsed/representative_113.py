#linear_scoring = cross_validation.cross_val_score(linear_regressor, data, target, scoring=scorer, 
#                                                  cv = 10)
#print 'mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std())

def testLinearRegression(df):
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_val_score
  from pandas.api.types import is_numeric_dtype
  import pandas

  linear_regression = LinearRegression()
  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  data = df[quantitativeColumns[:-1]]
  target = df[[quantitativeColumns[-1]]].values

  return pandas.DataFrame(cross_val_score(linear_regression, data, target, cv=10))

