def predict_test(loaded_dataset, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df = loaded_dataset
	reg = df[-1]
	X_train, X_test, y_train, y_test = df[-2]
	y_predict = reg.predict(X_test)
	new_df = pd.DataFrame()
	new_df['predicted'] = y_predict
	new_df['actual'] = list(y_test)
	res = {
		'output' : new_df.head(10).round(3).to_json(orient='table'),
		'result' : "predicted vs. actual test done, see Output Data Frame Tab.",
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.head(10).round(3))
	return res
res = predict_test(self.intermediate_df, description, method)