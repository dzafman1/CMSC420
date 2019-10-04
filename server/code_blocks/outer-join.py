def outer_join(loaded_dataset, intermediate_df, description, method):
	df1 = loaded_dataset

	import panas as pd
	import numpy as np

	res = {
		'output': df1.head(10).to_json(orient='table'),
		'result': df1.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}

	intermediate_df.append(df1.head(10))
	return res

res = outer_join(self.current_df, self.intermediate_df, description, method)