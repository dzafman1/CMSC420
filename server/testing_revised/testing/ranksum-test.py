def rank_sum(loaded_dataset, intermediate_df, description, method):
	import itertools
	from scipy import stats
	import pandas as pd
	
	current_df = loaded_dataset
	if not isinstance(current_df, pd.DataFrame): 
		current_df = current_df.to_frame()
	
	numerical_df = current_df.select_dtypes(include='number')

	if (numerical_df.empty == True):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	res_df = pd.DataFrame(columns=(numerical_df.columns), index=(numerical_df.columns))
	for col1, col2 in itertools.combinations(numerical_df, 2):
		z_stat, p_val = stats.ranksums(numerical_df[col1], numerical_df[col2])
		res_df[col1][col2] = p_val
		res_df[col2][col1] = p_val
	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : res_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(res_df.round(3))
	return res

