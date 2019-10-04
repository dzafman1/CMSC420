def predict_test(loaded_dataset, intermediate_df, description, method):

	import pandas as pd

	df = loaded_dataset
	print ("DF ROWS: \n", df.shape[0])
	if (df.shape[0] < 2):
		res = {
			'output': "Dataframe has less than two rows", 
			'result': "Dataframe has less than two rows", 
			'description' : "Dataframe has less than two rows",
			'type' : "error"
		}
		return res

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

