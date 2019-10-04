def demo_log_space(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset	
	
	from pandas.api.types import is_numeric_dtype
	from sklearn.model_selection import train_test_split
	import numpy as np
	import pandas as pd
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if len(quantitativeColumns) == 0:
		res = {
			'output': "Dataframe needs numeric values",
			'result': "Dataframe needs numeric values",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}
		return res

	X = df[quantitativeColumns[:-1]]
	y = df[[quantitativeColumns[-1]]].values.ravel()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	data = {'svc__C': np.logspace(-3, 2, 6), 'svc__gamma': np.logspace(-3, 2, 6) / X_train.shape[0]}
	res = {
		'output': pd.DataFrame(data).head(10).to_json(orient='table'),
		'result': pd.DataFrame(data).head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(data).head(10))
	return res

res = demo_log_space(self.current_df, self.intermediate_df, description, method)
