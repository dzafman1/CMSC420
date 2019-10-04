def corr_heatmap(loaded_dataset, intermediate_df, description, method):
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

	corr = df.select_dtypes(include='number').corr()
	image_list = []
	plt.clf()
	
	seaborn.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,annot=True, annot_kws={"size": 8}, square=True)
	save_bytes_image(image_list)
	res = {
		'output' : corr.round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(corr.round(3))
	return res
