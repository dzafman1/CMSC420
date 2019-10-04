def corr(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	numerical_df = df.select_dtypes(include='number')
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

# Pass
# df = pd.DataFrame({'a': [1, 2, 1, 2], 'b': [3, 2, 1, 3],  'c': [1.0, 2.0] * 2})

# Pass: should only compute using column c
# df = pd.DataFrame({'a': ["x", "y", "z"], 'b': ["w", "z", "u"], 'c': [4, 5, 6]})

# r = corr(df, [], "", "")
# print(r)
# print("\n")
# print(df)
