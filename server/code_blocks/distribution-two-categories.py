import pandas as pd
import numpy as np
import itertools
import io
import base64
import matplotlib.pyplot as plt
import seaborn as seaborn

def save_bytes_image(image_list):
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	image_list.append(base64.b64encode(bytes_image.getvalue()))
	bytes_image.seek(0)

def dist_two_categories(loaded_dataset, intermediate_df, description, method):
	image_list = []
	clean_dataset = pd.DataFrame()
	df = loaded_dataset
	
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset
	for col in df.columns:
		if df[col].dtype == 'object' or df[col].value_counts().count() <= 20:
			clean_dataset[col] = df[col].astype('category')
	
	category_df = clean_dataset.select_dtypes(include='category')

	if category_df.empty == True: 
		res = {
			'output': "Dataframe has no categorical values", 
			'result' : "Dataframe has no categorical values",
			'description' : "Dataframe has no categorical values",
			'type' : "error"
		}
		print (res['output'])
		return res

	count = 0
	for col1, col2 in itertools.combinations(category_df.columns, 2):
		if category_df[col1].value_counts().count() <= 10 and 			category_df[col2].value_counts().count() <= 5:
			seaborn.catplot(col1, data=category_df, hue=col2, kind='count')
			save_bytes_image(image_list)
			count += 1
			if count >= 5:
				break
	res = {
		'output' : category_df.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(category_df.head(10).round(3))
	print (category_df.head(10).round(3))
	return res

# df = pd.DataFrame(np.random.uniform(low=False, high=True, size=(29, 10)), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'])

# df = pd.DataFrame({'a': [0, 0] * 5, 'b': ["is", "kartik"] * 5,  'c': ["s", "krishnan"] * 5})
# print (df)

res = dist_two_categories(self.current_df, self.intermediate_df, description, method)