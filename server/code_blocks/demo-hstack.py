def demo_hstack(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	from pandas.api.types import is_numeric_dtype
	import numpy as np
	import pandas as pd
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if len(quantitativeColumns) == 0:
		raise ValueError('no numeric type found')

	x = df[quantitativeColumns[0]].values.ravel()
	y = df[quantitativeColumns[1]].values.ravel()
	x1 = 1 / x
	y1 = 1 / y
	x2, y2 = np.dot(np.random.uniform(size=(2, 2)), np.random.normal(size=(2, len(x))))
	u = np.hstack([x1, x2])
	v = np.hstack([y1, y2])
	res = {
		'output': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
		'result': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame([u, v]).head(10))
	return res

res = demo_hstack(self.current_df, self.intermediate_df, description, method)