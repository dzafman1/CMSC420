def dist_quant_category(loaded_dataset, intermediate_df, description, method):
	import pandas as pd 
	import seaborn
	import matplotlib.pyplot as plt

	image_list = []
	clean_dataset = pd.DataFrame()
	df = loaded_dataset
	for col in loaded_dataset.columns:
		if df[col].dtype == 'object' or df[col].value_counts().count() <= 20:
			clean_dataset[col] = df[col].astype('category')
		else:
			clean_dataset[col] = df[col]

	numerical_cols = clean_dataset.select_dtypes(include='number').columns
	category_cols = clean_dataset.select_dtypes(include='category').columns
	count = 0
	for col1, col2 in zip(numerical_cols, category_cols):
		print(clean_dataset[col2].value_counts().count())
		if clean_dataset[col2].value_counts().count() <= 10:
			seaborn.catplot(col2, col1, data=clean_dataset)
			save_bytes_image(image_list)
			count+=1
			if count >= 5:
				break
	res = {
		'output' : clean_dataset.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(clean_dataset.head(10).round(3))
	return res

res = dist_quant_category(self.current_df, self.intermediate_df, description, method)
