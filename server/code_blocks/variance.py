def variance(loaded_dataset, intermediate_df, description, method):
	try: 
		new_df = loaded_dataset.var()
	except Exception as e: 
		res = {
			'output': str(e), 
			'result': str(e), 
			'description' : str(e),
			'type' : "error"
		}
		return res

	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : new_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.round(3))
	return res

res = variance(self.current_df, self.intermediate_df, description, method)