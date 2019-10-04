import pandas as pd
import numpy as np


def drop_rows(loaded_dataset, intermediate_df, description, method):

	df = loaded_dataset
	if df.isnull().values.any() == False: 
		res = {
			'output': "Dataframe has no rows with NaN entries", 
			'result' : "Dataframe has no rows with NaN entries",
			'description' : "Dataframe has no rows with NaN entries",
			'type' : "error"
		}
		print (res['output'])
		return res

	new_df = loaded_dataset.dropna()



	# if len(intermediate_df) != 0:
	# 	new_df = intermediate_df[-1].dropna()
	# else:
	# 	new_df = loaded_dataset.dropna()
	
	res = {
		'output' : new_df.head(10).round(3).to_json(orient='table'),
		'result' : new_df.describe().round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.head(10).round(3))
	print (new_df.describe().round(3))
	return res

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(29, 10)), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'])

# df = pd.DataFrame({'a': [0,0] * 5, 'b': ["is", "kartik"] * 5,  'c': ["s", "krishnan"] * 5})
# print (df)
res = drop_rows(self.current_df, self.intermediate_df, description, method)