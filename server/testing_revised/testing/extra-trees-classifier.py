def extra_trees_classifier(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import train_test_split
	from pandas.api.types import is_numeric_dtype
	from sklearn.metrics import accuracy_score
	import pandas as pd

	ETC = ExtraTreesClassifier()
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
	if (len(quantitativeColumns) == 0):
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
	ETC = ExtraTreesClassifier()
	ETC.fit(X_train, y_train)
	prediction = ETC.predict(X_test)
	scores = cross_val_score(ETC,X_train,y_train,cv=2)

	res = {
		'output': pd.DataFrame(scores).to_json(orient='table'),
		'result': pd.DataFrame(scores).to_json(orient='table'),
		'description' : description,
		'type': method
	}

	intermediate_df.append(pd.DataFrame(scores))
	return res

