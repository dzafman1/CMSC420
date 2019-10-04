def compute_covariance_matrix(loaded_dataset, intermediate_df, description, method):

	import pandas as pd
	import numpy as np
	import sys	

	df = loaded_dataset.select_dtypes(include='number')

	if df.empty == True: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : 'error'
		}
		return res

	#convert numerical columns in df to matrix representation
	df_matrix = df.as_matrix()
		
	data = {"CovMeanDot":[]}
	covariance = np.cov(df_matrix)
	mean = np.mean(df_matrix, axis=0)
	# inv = np.linalg.inv(covariance)

	#checks if covariance matrix is singular or not
	if not (np.linalg.cond(covariance) < 1/sys.float_info.epsilon):
    	#handle it
		res = {
			'output': "Matrix is singular", 
			'result': "Matrix is singular", 
			'description' : "Matrix is singular",
			'type' : 'error'
		}
		print (res['output'])
		return res
	else: 
		inv = np.linalg.inv(covariance)

	dot = np.dot(np.dot(mean, inv), mean)
	data["CovMeanDot"].append(dot)
	res = {
		'output': pd.DataFrame(data).to_json(orient='table'),
		'result': pd.DataFrame(data).to_json(orient='table'),
		'description' : description,
		'type': method
	}

	intermediate_df.append(pd.DataFrame(data))
	return res

res = compute_covariance_matrix(self.current_df, self.intermediate_df, description, method)
