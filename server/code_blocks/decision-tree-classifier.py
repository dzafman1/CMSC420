def decision_tree_classifier(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.model_selection import train_test_split
	from pandas.api.types import is_numeric_dtype
	from sklearn.metrics import accuracy_score
	import pandas as pd

	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
	X = df[quantitativeColumns[:-1]]
	y = df[[quantitativeColumns[-1]]].values.ravel()
	classifier = DecisionTreeClassifier()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	classifier.fit(X_train, y_train)
	prediction = classifier.predict(X_test)
	data = {'accuracyScore': []}
	data['accuracyScore'].append(accuracy_score(y_test, prediction, normalize=False))
	res = {
		'output': pd.DataFrame(data).head(10).to_json(orient='table'),
		'result': pd.DataFrame(data).head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(data).head(10))
	return res
res = decision_tree_classifier(self.current_df, self.intermediate_df, description, method)

# Pass: accuracy score of 2
# df = pd.DataFrame({'a': [1, 2] * 3, 'b': [3, 3] * 3,  'c': [1.0, 2.0] * 3})

# Pass: accuracy of 3 (testing different dimensions)
# df = pd.DataFrame({'a': [1, 2] * 4, 'b': [3, 3] * 4,  'c': [1.0, 2.0] * 4})
# r = decision_tree_classifier(df, [], "", "")
# print(r)
# print(df)
