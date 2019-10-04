def word_to_vec(loaded_dataset, intermediate_df, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df= loaded_dataset
	res_df = calcWordVec(df)
	res = {
		'output': res_df.head(10).to_json(orient='table'),
		'result': res_df.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(res_df.head(10))
	return res
res = word_to_vec(self.current_df, self.intermediate_df, description, method)