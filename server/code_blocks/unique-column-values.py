def unique_column_values(loaded_dataset, intermediate_df, description, method):
	test = {}

	import pandas as pd

	df = loaded_dataset
	alt_df = df.select_dtypes(include='number')

	if (alt_df.empty == True):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res
	
	for column in alt_df:
		test[column] = alt_df[column].dropna().unique()
	res = {
		'output': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
		'result': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10))
	return res

res = unique_column_values(self.current_df, self.intermediate_df, description, method)