def quantitative_bar_plot(loaded_dataset, intermediate_df, description, method):
	from pandas.api.types import is_numeric_dtype
	import matplotlib.pyplot as plt
	import io 
	import base64
	

	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
	
	df = loaded_dataset
	image_list = []
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if (len(quantitativeColumns) == 0):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res
	
	x = df[quantitativeColumns[0]].values.ravel()
	y = df[[quantitativeColumns[1]]].values.ravel()
	plt.figure()
	plt.title("Plot Bar")
	plt.bar(range(len(x)), y, align="center")
	plt.xticks(range(len(x)), rotation=90)
	plt.xlim([-1, len(x)])
	save_bytes_image(image_list)
	res = {
		'output': df.head(10).to_json(orient='table'),
		'result': image_list,
		'description' : description,
		'type': method
	}
	intermediate_df.append(df.head(10))
	return res

res = quantitative_bar_plot(self.current_df, self.intermediate_df, description, method)