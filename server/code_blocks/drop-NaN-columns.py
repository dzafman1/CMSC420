import pandas as pd
import numpy as np

def drop_cols(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset

	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	

	dropped_columns = []
	df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]

	#df2 holds the columns that have more than 30% NaN entries - if empty - algo should be run
	
	
	for c in df.columns:
		if c not in df2.columns:
			dropped_columns.append(c)

	if len(dropped_columns) == 0: 
		res = {
			'output': "Dataframe has less than 30% NaN entries", 
			'result' : "Dataframe has less than 30% NaN entries",
			'description' : "Dataframe has less than 30% NaN entries",
			'type' : "error"
		}
		print (res['output'])
		return res
	loaded_dataset = df2
	res = {
		'output' : df.describe().round(3).to_json(orient='table'),
		'result' : df.describe().round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(df.describe().round(3))
	print (res['output'])
	return res

df = pd.DataFrame(np.random.uniform(low=False, high=True, size=(29, 10)), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'])

# df = pd.DataFrame({'a': [0, 0] * 5, 'b': ["is", "kartik"] * 5,  'c': ["s", "krishnan"] * 5})
# print (df)

res = drop_cols(df, [], "", "")