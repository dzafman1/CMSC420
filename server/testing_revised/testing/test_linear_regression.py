def test_linear_regression(loaded_dataset, intermediate_df, description, method):
	import pandas as pd
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import cross_val_score
	from pandas.api.types import is_numeric_dtype

	linear_regression = LinearRegression()
	df = loaded_dataset

	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if (len(quantitativeColumns) == 0):
		res = {
			'output': "Dataframe has no numeric values",
			'result': "Dataframe has no numeric values",
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	data = df[quantitativeColumns[:-1]]
	target = df[[quantitativeColumns[-1]]].values

	res = {
	'output': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
	'result': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
	'description' : description,
	'type': method
	}

	intermediate_df.append(pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)))
	return res
