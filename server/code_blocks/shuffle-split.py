def shuffle_split(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	numerical_df = df.select_dtypes(include='number')

	if (numerical_df.empty == True):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res
	
	features = numerical_df[numerical_df.columns[0:3]]
	predicted_variables = numerical_df[numerical_df.columns[-1]]
	
	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(features, predicted_variables, test_size=0.2,random_state=100)
	new_df = (X_train, X_test, y_train, y_test)
	res = {
		'output' : X_train.head(10).round(3).to_json(orient="table"),
		'result' : "split into training and testing set",
		'description' : description,
		'type' : method
	}
	intermediate_df.append(X_train.head(10).round(3))
	return res

res = shuffle_split(self.current_df, self.intermediate_df, description, method)