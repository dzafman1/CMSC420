import pandas as pd
import numpy as np


#test this with non umeric values - then check if needed
def mean(loaded_dataset, intermediate_df, description, method):

	df = loaded_dataset.select_dtypes(include='number')

	if df.empty == True: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		print (res['output'])
		return res

	new_df = pd.DataFrame(df.mean(), columns=['mean'])
	

	# if len(intermediate_df) != 0:
	# 	new_df = pd.DataFrame(intermediate_df[-1].mean(), columns=['mean'])
	# else:
	# 	new_df = pd.DataFrame(loaded_dataset.mean(), columns=['mean'])

	print(new_df)
	res = {
		'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
		'result' : new_df.round(3).to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(new_df.round(3))
	print (res['output'])
	return res

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

df = pd.DataFrame({'a': ["hi", "hi"] * 5, 'b': [0, 1] * 5,  'c': ["", ""] * 5})

res = mean(df, [], "", "")