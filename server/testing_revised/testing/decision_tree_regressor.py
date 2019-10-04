

def decision_tree_regressor(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	from sklearn.tree import DecisionTreeRegressor
	from pandas.api.types import is_numeric_dtype
	import pandas as pd
	import numpy as np

	tree_reg1 = DecisionTreeRegressor(random_state=42)
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if df.isnull().any().any() == True:
		res = {
			'output': "Dataframe needs numeric values",
			'result': "Dataframe needs numeric values",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}
		return res

	if len(quantitativeColumns) == 0:
		res = {
			'output': "Dataframe needs numeric values",
			'result': "Dataframe needs numeric values",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}
		return res

	X = df[quantitativeColumns[:-1]]
	y = df[[quantitativeColumns[-1]]]
	tree_reg1.fit(X, y)
	y_pred1 = tree_reg1.predict(X)
	out_df = X.copy()
	out_df["Expected-"+quantitativeColumns[-1]] = y
	out_df["Predicted-"+quantitativeColumns[-1]] = y_pred1
	res = {
		'output': out_df.head(10).to_json(orient='table'),
		'result': out_df.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(out_df.head(10))
	return res
