def variance(loaded_dataset, description, method):
	new_df = None
	if len(intermediate_df) != 0:
		new_df = intermediate_df[-1].var()
	else:
		df_matrix = loaded_dataset.var()
	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : new_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	self.intermediate_df.append(new_df.round(3))
	return res
res = variance(self.current_df, description, method)