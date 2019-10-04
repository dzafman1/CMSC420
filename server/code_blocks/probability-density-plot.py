def probability_density_plot(loaded_dataset, intermediate_df, description, method):
	df = None
	if len(intermediate_df) != 0:
		df = intermediate_df[-1]
	else:
		df = loaded_dataset
	image_list = []
	from scipy.stats import chi2
	from pandas.api.types import is_numeric_dtype
	data = {'rv':[], 'pdf':[]}
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
	x = df[quantitativeColumns[0]].values.ravel()
	for k in [1, 2]:
		rv = chi2(k)
		pdf = rv.pdf(x)
		data['rv'].append(rv)
		data['pdf'].append(pdf)
		plt.plot(x, pdf, label="$k=%s$" % k)
	plt.legend()
	plt.title("PDF ($\chi^2_k$)")
	save_bytes_image(image_list)
	output_df = pd.DataFrame(data)
	res = {
		'output': output_df.head(10).to_json(orient='table'),
		'result': image_list,
		'description' : description,
		'type': method
	}
	intermediate_df.append(output_df.head(10))
	return res
res = probability_density_plot(self.current_df, self.intermediate_df, description, method)