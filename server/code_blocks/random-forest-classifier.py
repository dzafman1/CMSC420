def random_forest_classifier(loaded_dataset, intermediate_df, description, method):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import cross_val_score
	from pandas.api.types import is_numeric_dtype
	import pandas as pd
	
	df = loaded_dataset.select_dtypes(include='number')
	
	forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=17)
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if (len(quantitativeColumns) == 0):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	X_train = df[quantitativeColumns[:-1]]
	y_train = df[[quantitativeColumns[-1]]].values.ravel()
	res = {
		'output': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
		'result': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)))
	return res

res = random_forest_classifier(self.current_df, self.intermediate_df, description, method)