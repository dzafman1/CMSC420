def scatterplot_regression(loaded_dataset, intermediate_df, description, method):
	import matplotlib.pyplot as plt
	import seaborn
	import itertools
	import io
	import base64
	
	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
	
	df = loaded_dataset
	numerical_df = df.select_dtypes(include='number')

	if (numerical_df.empty == True):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res
	
	image_list = []
	count = 0
	for col1, col2 in itertools.combinations(numerical_df, 2):
		plt.clf()
		seaborn.regplot(df[col1], df[col2])
		save_bytes_image(image_list)
		plt.show()
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

