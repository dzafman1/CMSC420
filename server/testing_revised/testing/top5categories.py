def top5cat(loaded_dataset, intermediate_df, description, method):
	category_df = loaded_dataset.select_dtypes(include='object')

	if (category_df.empty == True):
		res = {
			'output': "Dataframe has no category values", 
			'result': "Dataframe has no category values", 
			'description' : "Dataframe has no category values",
			'type' : "error"
		}
		return res

	for col in category_df:
		samples = category_df[col].value_counts().head(5)
		intermediate_df.append(samples.round(3))
		return {
			'output' : samples.round(3).to_json(orient='table'),
			'result' :samples.round(3).to_json(orient='table'),
			'description' : description,
			'type' : method
		}

