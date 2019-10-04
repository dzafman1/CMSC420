def cat_count(loaded_dataset, intermediate_df, description, method):

	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as seaborn
	import io
	import base64
	from pandas.api.types import is_string_dtype

	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)

	df = loaded_dataset
	category_df = df.select_dtypes(include='object')
	#print(category_df)

	if category_df.empty == True:
		res = {
			'output': "Dataframe contained incorrect values",
			'result' : "Dataframe contained incorrect values",
			'description' : "Dataframe contained incorrect values",
			'type': "error"
		}
		return res

	image_list = []

	category_df = category_df.dropna(axis='columns')
	#print("new DF \n", category_df)
	for col in category_df:
		#check to make sure 'object' type is actually a string - assuming this is what is needed
		if is_string_dtype(category_df[col]) != True:
				res = {
				'output': "Illegal dataframe value",
				'result' : "Illegal dataframe value",
				'description' : "Illegal dataframe value",
				'type': 'error'
				}
				return res

		if category_df[col].value_counts().count() <= 20:
			seaborn.catplot(x=col, data=category_df, alpha=0.7, kind='count')
			save_bytes_image(image_list)
	res = {
		'output' : category_df.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(category_df.head(10).round(3))
	return res
