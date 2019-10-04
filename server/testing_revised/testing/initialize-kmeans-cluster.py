def initialize_kmeans_cluster(loaded_dataset, intermediate_df, description, method):
	df= loaded_dataset
	try:
		res_df = initializeClustersForKmeans(df)
	except Exception as e:
		res = {
			'output': str(e),
			'result': str(e),
			'description' : str(e),
			'type': 'error'
		}

		return res

	res = {
		'output': df.head(10).to_json(orient='table'),
		'result': res_df.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(df.head(10))
	return res

