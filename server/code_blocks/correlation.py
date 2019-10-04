def corr(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	numerical_df = df.select_dtypes(include='number')
	if (numerical_df.empty == True):
		res = {
			'result': "Dataframe needs numeric values",
			'output': "Dataframe needs numeric values",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}

		return res
	
	correlations = numerical_df.corr()
	res = {
		'result' : correlations.round(3).to_json(orient='table'),
		'output' : correlations.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	
	intermediate_df.append(correlations.round(3))
	return res
res = corr(self.loaded_dataset, self.intermediate_df, description, method)