def bootstrap(loaded_dataset, intermediate_df, description, method):
	#get columns that have numerical values

	#check dtype of all values in dataframe

	import pandas as pd
	import numpy as np

	current_df = loaded_dataset.select_dtypes('number')

	if current_df.empty == True:
		res = {
			'output': "Dataframe has no numeric values", 
			'result' : "Dataframe has no numeric values",
			'description' : "Dataframe has no numeric values",
			'type' : 'error'
		}
		return res

	from sklearn.utils import resample
	mean = []
	statistics = pd.DataFrame()

	for i in range(0, 1000):
		boot = resample(current_df, replace=True, n_samples=int(0.5 * len(current_df.index)))
		mean.append(boot.mean().to_dict())
	for key in mean[0]:
		curr_list = [item[key] for item in mean]
		alpha = 0.95
		p = (1.0-alpha) * 100
		lower = np.percentile(curr_list, p)
		p = alpha * 100
		upper = np.percentile(curr_list, p)
		statistics[key] = [str(round(lower, 3)) + '-' + str(round(upper, 3))]
	res = {
		'output' : current_df.head(10).round(3).to_json(orient='table'),
		'result' : statistics.to_json(orient='table'),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(current_df.head(10).round(3))
	return res

res = bootstrap(self.current_df, self.intermediate_df, description, method)
