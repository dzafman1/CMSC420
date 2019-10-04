import pandas as pd
import numpy as np

def eval_model_predictions(loaded_dataset, intermediate_df, description, method):
	
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
	
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1].select_dtypes(include='number')
	# else:
	# 	df = loaded_dataset.select_dtypes(include='number')


	predictions = df.iloc[:,-1].values
	labels = df.iloc[:,-2].values
	res = {
		'output': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
		'result': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(np.equal(predictions,labels)))
	# print (res['output'])
	return res
# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(29, 10)), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'])

# df = pd.DataFrame({'a': [0, 0] * 5, 'b': [0, 0] * 5,  'c': [False, True] * 5})

res = eval_model_predictions(self.current_df, self.intermediate_df, description, method)