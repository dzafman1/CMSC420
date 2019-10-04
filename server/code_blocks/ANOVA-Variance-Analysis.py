def anova_variance(loaded_dataset, intermediate_df, description, method):
	import pandas as pd
	import numpy as np
	from pandas.api.types import is_string_dtype
	from pandas.api.types import is_numeric_dtype

	current_df = loaded_dataset
	
	print ("ANOVA VARIANCE INPUT DF")
	print (current_df)

	category_cols = current_df.select_dtypes(include='object').columns
	numerical_cols = current_df.select_dtypes(include='number').columns

	res_df = pd.DataFrame(columns=category_cols, index=numerical_cols)

	if len(category_cols) == 0 or len(numerical_cols) == 0:

			res = {
				'output': "Dataframe contained incorrect values", 
				'result' : "Dataframe contained incorrect values",
				'description' : "Dataframe contained incorrect values",
				'type' : 'error'
			}
			print (res['output'])
			return res

	for num_col in numerical_cols:
		#check to make sure num_col has all numeric values:
		if is_numeric_dtype(current_df[num_col]) != True:
			res = {
				'output': "Illegal dataframe value num_col", 
				'result' : "Illegal dataframe value",
				'description' : "Illegal dataframe value",
				'type' : 'error'
			}
			print (res['output'])
			return res
		
		for cat_col in category_cols:
			#assuming this is checking for strings 
			if is_string_dtype(current_df[cat_col]) != True:
				res = {
				'output': "Illegal dataframe value cat_col", 
				'result' : "Illegal dataframe value",
				'description' : "Illegal dataframe value",
				'type' : 'error'
				}

				print (res['output'])
				return res

			if current_df[cat_col].value_counts().count() <= 10:
				groups = current_df.groupby(cat_col).groups.keys()
				print groups
				print current_df[current_df[cat_col] == groups[0]][num_col]
				if len(groups) >= 3:
					f_val, p_val = stats.f_oneway(current_df[current_df[cat_col] == groups[0]][num_col], current_df[current_df[cat_col] == groups[1]][num_col], current_df[current_df[cat_col] == groups[2]][num_col])
					res_df[cat_col][num_col] = p_val

	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : res_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}

	intermediate_df.append(res_df.round(3))

	return res

res = anova_variance(self.current_df, self.intermediate_df, description, method)
