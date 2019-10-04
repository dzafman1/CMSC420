def corr_heatmap(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	import matplotlib.pyplot as plt
	import seaborn
	corr = df.select_dtypes(include='number').corr()
	image_list = []
	plt.clf()
	seaborn.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,annot=True, annot_kws={"size": 8}, square=True)
	save_bytes_image(image_list)
	#plt.show()
	res = {
		'output' : corr.round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(corr.round(3))
	return res
res = corr_heatmap(self.current_df, description, method)

# df = pd.DataFrame({'a': [1, 2, 1], 'b': [3, 2, 1],  'c': [1.0, 2.0, 3.0]})
# r = corr_heatmap(df, [], "", "")
# print(r)
# print("\n")
# print(df)
