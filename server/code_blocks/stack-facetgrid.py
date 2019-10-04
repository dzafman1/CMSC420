def stack_ftgrid(loaded_dataset, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df = loaded_dataset
	numerical_cols = df.select_dtypes(include='number').columns
	category_cols = df.select_dtypes(include='object').columns
	image_list = []
	for cat_var in category_cols:
		if df[cat_var].value_counts().count() <= 5:
			for num_var in numerical_cols:
				plt.clf()
				fig = seaborn.FacetGrid(df,hue=cat_var)
				fig.map(seaborn.kdeplot,num_var,shade=True)
				oldest = df[num_var].max()
				fig.set(xlim=(0, oldest))
				fig.add_legend()
				save_bytes_image(image_list)
				if len(image_list) >= 5:
					break
	res = {
		'output' : df.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(df.head(10).round(3))
	return res
res = stack_ftgrid(self.current_df, description, method)
self.intermediate_df.append(self.current_df)