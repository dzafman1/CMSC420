def top5cat(loaded_dataset, description, method):
	category_df = None
	if len(intermediate_df) != 0:
		category_df = intermediate_df[-1].select_dtypes(include='object')
	else:
		category_df = loaded_dataset.select_dtypes(include='object')
	for col in category_df:
		samples = category_df[col].value_counts().head(5)
		res = {
			'output' : samples.round(3).to_json(orient='table'),
			'result' :samples.round(3).to_json(orient='table'),
		'description' : description,
			'type' : method
		}
		break
	intermediate_df.append(samples.round(3))
	return res
res = top5cat(self.current_df, description, method)
self.intermediate_df.append(self.current_df)