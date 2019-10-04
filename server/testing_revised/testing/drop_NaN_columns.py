

def drop_cols(loaded_dataset, intermediate_df, description, method):
	import pandas as pd
	import numpy as np

	df = loaded_dataset
	dropped_columns = []
	df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]

	#df2 holds the columns that have more than 30% NaN entries - if empty - algo should be run
	for c in df.columns:
		if c not in df2.columns:
			dropped_columns.append(c)

	if len(dropped_columns) == 0: 
		res = {
			'output': df.describe().round(3).to_json(orient='table'), 
			'result' : df.describe().round(3).to_json(orient='table'),
			'description' : "Dataframe has less than 30% NaN entries",
			'type' : "error"
		}
		return res
	loaded_dataset = df2
	res = {
		'output' : df2.describe().round(3).to_json(orient='table'),
		'result' : df2.describe().round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(df2.describe().round(3))
	return res


