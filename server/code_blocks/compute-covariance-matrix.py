import pandas as pd
import numpy as np
import sys



def compute_covariance_matrix(loaded_dataset, intermediate_df, description, method):

	df = loaded_dataset.select_dtypes(include='number')

	if df.empty == True: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values"
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
			'description' : "Matrix is singular"
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
	# print (res['output'])
	return res

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

# df = pd.DataFrame({'a': ["", ""] * 2, 'b': ["", ""] * 2,  'c': ["", ""] * 2})

res = compute_covariance_matrix(self.current_df, self.intermediate_df, description, method)