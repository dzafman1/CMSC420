def compute_percentiles_range(loaded_dataset, intermediate_df, description, method):
	from pandas.api.types import is_numeric_dtype
	import pandas as pd
	import numpy as np
	
	lowerPercentile = 5
	upperPercentile = 95
	df = loaded_dataset
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if len(quantitativeColumns) == 0:
		res = {
			'output': "Dataframe needs numeric values",
			'result': "Dataframe needs numeric vavlues",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}
		return res

	data = {"Percentile"+str(lowerPercentile):[],"Percentile"+str(upperPercentile):[],"columnName":quantitativeColumns}
	for c in quantitativeColumns:
		data["Percentile"+str(lowerPercentile)].append(np.percentile(df[[c]],lowerPercentile))
		data["Percentile"+str(upperPercentile)].append(np.percentile(df[[c]],upperPercentile))
	res = {
		'output': pd.DataFrame(data).to_json(orient='table'),
		'result': pd.DataFrame(data).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(data))
	return res

