
#test this with non umeric values - then check if needed
def mean(loaded_dataset, intermediate_df, description, method):

	df = loaded_dataset.select_dtypes(include='number')

	import pandas as pd
	import numpy as np

	if df.empty == True: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	new_df = pd.DataFrame(df.mean(), columns=['mean'])

	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : new_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.round(3))
	return res

res = mean(self.current_df, self.intermediate_df, description, method)
