def stack_ftgrid(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset

	import matplotlib.pyplot as plt
	import seaborn
	import io
	import base64

	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
	

	numerical_cols = df.select_dtypes(include='number').columns
	category_cols = df.select_dtypes(include='object').columns

	if (len(numerical_cols) == 0 OR len(category_cols) == 0):
		res = {
			'output': "Dataframe has no numeric or cateogry values", 
			'result': "Dataframe has no numeric or category values", 
			'description' : "Dataframe has no numeric or category values",
			'type' : "error"
		}
		return res
	
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

