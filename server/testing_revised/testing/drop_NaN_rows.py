def drop_rows(loaded_dataset, intermediate_df, description, method):
	import pandas as pd
	import numpy as np
	
	df = loaded_dataset
	if df.isnull().values.any() == False: 
		res = {
			'output': df.head(10).to_json(orient='table'), 
			'result' : df.head(10).to_json(orient='table'),
			'description' : "Dataframe has no rows with NaN entries",
			'type' : "error"
		}
		return res

	new_df = loaded_dataset.dropna()

	res = {
		'output' : new_df.head(10).round(3).to_json(orient='table'),
		'result' : new_df.describe().round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.head(10).round(3))
	return res

