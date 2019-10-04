def scatterplot_regression(loaded_dataset, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df = loaded_dataset
	numerical_df = df.select_dtypes(include='number')
	image_list = []
	count = 0
	for col1, col2 in itertools.combinations(numerical_df, 2):
		plt.clf()
		seaborn.regplot(df[col1], df[col2])
		save_bytes_image(image_list)
		count+=1
		if count >= 5:
			break
	res = {
		'output' : numerical_df.head(10).round(3).to_json(orient='table'),

		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(numerical_df.head(10).round(3))
	return res
res = scatterplot_regression(self.current_df, description, method)
self.intermediate_df.append(self.current_df)