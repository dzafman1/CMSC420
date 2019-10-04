import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seaborn
import io
import base64
from pandas.api.types import is_string_dtype


def save_bytes_image(image_list):
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    image_list.append(base64.b64encode(bytes_image.getvalue()))
    bytes_image.seek(0)

def cat_count(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	category_df = df.select_dtypes(include='object')

	if category_df.empty == True: 
		res = {
			'output': "Dataframe contained incorrect values", 
			'result' : "Dataframe contained incorrect values",
			'description' : "Dataframe contained incorrect values"
		}
		print(res['output'])
		return res
		
	image_list = []

	for col in category_df:
		#check to make sure 'object' type is actually a string - assuming this is what is needed
		if is_string_dtype(category_df[col]) != True:
				res = {
				'output': "Illegal dataframe value", 
				'result' : "Illegeal dataframe value",
				'description' : "Illegal dataframe value"
				}
				print (res['output'])
				return res
		
		if category_df[col].value_counts().count() <= 20:
			seaborn.catplot(x=col, data=category_df, alpha=0.7, kind='count')
			save_bytes_image(image_list)
	res = {
		'output' : category_df.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(category_df.head(10).round(3))
	print (res['output'])
	return res

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

# df = pd.DataFrame({'a': ["hi", ""] * 2, 'b': [False, False] * 2,  'c': [1, 2] * 2})

res = cat_count(self.current_df, self.intermediate_df, description, method)