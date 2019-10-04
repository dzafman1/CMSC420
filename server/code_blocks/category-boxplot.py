def cat_boxplot(loaded_dataset, intermediate_df, description, method):

	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as seaborn
	import io
	import base64
	
	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
		df = loaded_dataset

	numerical_cols = df.select_dtypes(include='number').columns
	category_cols = df.select_dtypes(include='object').columns

	if len(category_cols) == 0 or len(numerical_cols) == 0:
		res = {
			'output': "Dataframe contained incorrect values", 
			'result' : "Dataframe contained incorrect values",
			'description' : "Dataframe contained incorrect values",
			'type' : 'error'
		}
		return res

	image_list = []
	for cat_var in category_cols:
		if df[cat_var].value_counts().count() <= 5:
			for num_var in numerical_cols:
				plt.clf()
				ax = seaborn.boxplot(x=cat_var, y=num_var, data=df)
				plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
				plt.xticks(rotation=45)
				save_bytes_image(image_list)
				if len(image_list) >= 5:
					break
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

res = cat_boxplot(self.current_df, self.intermediate_df, description, method)