def plot(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	image_list = []

	import matplotlib.gridspec as gridspec
	import numpy as np
	import io
	import base64
	import matplotlib.pyplot as plt

	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
	
	samples = dict()
	alt_df = df.select_dtypes(include='number')

	if (alt_df.empty == True):
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		return res

	h, w = alt_df.shape
	for a in np.arange(h):
		samples[a] = ((alt_df.iloc[[a]].values).ravel())[:4]
		if len(samples) >= 5:
			break
	fig = plt.figure(figsize=(10, 10))
	gs = gridspec.GridSpec(1, len(samples))
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in samples.iteritems():
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(2, 2), cmap='Greys_r')
	save_bytes_image(image_list)
	res = {
		'output': df.head(10).to_json(orient='table'),
		'result': image_list,
		'description' : description,
		'type': method
	}
	intermediate_df.append(df.head(10))
	return res

