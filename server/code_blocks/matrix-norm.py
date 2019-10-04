import numpy.linalg as LA
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def matrix_norm(loaded_dataset, intermediate_df, description, method):

	df = loaded_dataset
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	nCols = [c for c in list(df) if is_numeric_dtype(df[c])]

	#if length is 0 that means no columns contained any numerical data
	if len(nCols) == 0: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		print (res['output'])
		return res


	data = {"columnName":[],"NumpyLinalgNorm":[]}
	for nc in nCols:
		data["columnName"].append(nc)
		data["NumpyLinalgNorm"].append(LA.norm(df[[nc]].values))
	res = {
		'output': pd.DataFrame(data).to_json(orient='table'),
		'result': pd.DataFrame(data).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(data))
	print (res['output'])
	return res

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

# df = pd.DataFrame({'a': ["hi", "hi"] * 5, 'b': ["", ""] * 5,  'c': ["", ""] * 5})

res = matrix_norm(self.current_df, self.intermediate_df, description, method)