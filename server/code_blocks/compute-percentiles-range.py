import pandas as pd
import numpy as np



def compute_percentiles_range(loaded_dataset, intermediate_df, description, method):
	from pandas.api.types import is_numeric_dtype
	
	lowerPercentile = 5
	upperPercentile = 95
	df = loaded_dataset

	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
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

res = compute_percentiles_range(self.current_df, self.intermediate_df, description, method)

# Pass
# df = pd.DataFrame({'a': [1, 2, 1, 2], 'b': [3, 2, 1, 3],  'c': [1.0, 2.0] * 2})

# Pass
# df = pd.DataFrame({'a': [1, 2, 1], 'b': [3, 2, 1],  'c': [1.0, 2.0, 3.0]})

# Non-numeric data
# df = pd.DataFrame({'a': ["x", "y"], 'b': ["w", "z"]})

# Error:  inconsistent dimensions
# df = pd.DataFrame({'a': [1, 2], 'b': [3, 2, 1, 3],  'c': [1.0, 2.0] * 2})

# Uncomment to run tests
# r = compute_percentiles_range(df, [], "", "")
# print(r['result'])
# print("\n")
# print(df)
