import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def save_bytes_image(image_list):
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	image_list.append(base64.b64encode(bytes_image.getvalue()))
	bytes_image.seek(0)

def dist_num(loaded_dataset, intermediate_df, description, method):
	image_list = []
	# df = None
	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	df = loaded_dataset

	#cols that have numerical values
	numerical_df = df.select_dtypes(include='number')

	#checks to see if input df has any numerical values
	if numerical_df.empty == True: 
		res = {
			'output': "Dataframe has no numerical values", 
			'result' : "Dataframe has no numerical values",
			'description' : "Dataframe has no numerical values",
			'type' : "error"
		}
		print (res['output'])
		return res

	count = 0
	for col in numerical_df:
		fig, ax = plt.subplots()
		ax.hist(numerical_df[col])
		plt.xlabel(col)
		plt.ylabel("Dist")
		plt.title('Histogram of ' + col)
		save_bytes_image(image_list)
		count += 1
		if count >= 5:
			break
	res = {
		'output' : numerical_df.head(10).round(3).to_json(orient='table'),
		'result' : image_list,
		'description' : description,
		'type' : method
	}
	intermediate_df.append(numerical_df.head(10).round(3))
	print (res['output'])
	return res




df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

df = pd.DataFrame({'a': [2, 1] * 5, 'b': [False, False] * 5,  'c': [False, False] * 5})

res = dist_num(self.current_df, self.intermediate_df, description, method)
