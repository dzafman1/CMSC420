import pandas as pd
import numpy as np
import io
import base64
import seaborn as seaborn
import matplotlib.pyplot as plt



def num_boxplot(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	numerical_cols = df.select_dtypes(include='number').columns

	if len(numerical_cols) == 0: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		print (res['output'])
		return res


	image_list = []
	for num_var in numerical_cols:
		plt.clf()
		ax = seaborn.boxplot(y=num_var, data=df)
		plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
		plt.xticks(rotation=45)
		save_bytes_image(image_list)
		if len(image_list) >= 5:
			break
	res = {
		'output' : df.select_dtypes(include='number').head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}

	intermediate_df.append(df.select_dtypes(include='number').head(10).round(3))
	print (res['output'])
	return res

def save_bytes_image(image_list):
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	image_list.append(base64.b64encode(bytes_image.getvalue()))
	bytes_image.seek(0)

# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

# df = pd.DataFrame({'a': ["hi", "hi"] * 5, 'b': ["", ""] * 5,  'c': ["", ""] * 5})

res = num_boxplot(self.current_df, self.intermediate_df, description, method)