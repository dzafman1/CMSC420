def dist_quant_category(loaded_dataset, intermediate_df, description, method):
	import pandas as pd 
	import seaborn
	import matplotlib.pyplot as plt

	def save_bytes_image(image_list):
		bytes_image = io.BytesIO()
		plt.savefig(bytes_image, format='png')
		image_list.append(base64.b64encode(bytes_image.getvalue()))
		bytes_image.seek(0)
	
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
	
	if (len(numerical_cols) == 0 or len(category_cols) == 0):
		res = {
			'output': "Dataframe needs numeric AND category values",
			'result': "Dataframe needs numeric AND category values",
			'description': "Dataframe needs numeric AND category values",
			'type' : 'error'
		}
		return res

	count = 0
	for col1, col2 in zip(numerical_cols, category_cols):
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
