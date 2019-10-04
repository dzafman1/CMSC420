def eval_model_predictions(loaded_dataset, intermediate_df, description, method):
	import pandas as pd
	import numpy as np

	df = loaded_dataset.select_dtypes(include='number')
	if df.empty == True: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	predictions = df.iloc[:,-1].values
	labels = df.iloc[:,-2].values
	res = {
		'output': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
		'result': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(np.equal(predictions,labels)))
	return res

res = eval_model_predictions(self.current_df, self.intermediate_df, description, method)