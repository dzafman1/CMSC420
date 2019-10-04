def des(loaded_dataset, intermediate_df, description, method): 
	descriptive_statistics = loaded_dataset.describe(include='all')
	res = {
		'result' : descriptive_statistics.round(3).to_json(orient='table'),
		'output' : descriptive_statistics.round(3).to_json(orient='table'),
		'description' : description,
		'type' : 'group-statistics'
	}
	intermediate_df.append(descriptive_statistics.round(3))
	return res

res = des(self.current_df, self.intermediate_df, description, method)