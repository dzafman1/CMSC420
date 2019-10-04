def plot(loaded_dataset, intermediate_df, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df_matrix = loaded_dataset
	image_list = []
	import matplotlib.gridspec as gridspec
	samples = dict()
	alt_df = df.select_dtypes(include='number')
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
res = plot(self.current_df, self.intermediate_df, description, method)