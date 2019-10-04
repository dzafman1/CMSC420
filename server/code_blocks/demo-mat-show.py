def demo_mat_show(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset
	image_list = []

	from pandas.api.types import is_numeric_dtype
	from sklearn.metrics import confusion_matrix
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt

	data = {'confusion': []}
	quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

	if len(quantitativeColumns) == 0:
		res = {
			'output': "Dataframe needs numeric values",
			'result': "Dataframe needs numeric values",
			'description': "Dataframe needs numeric values",
			'type' : 'error'
		}
		return res
	
	yy_test = df[quantitativeColumns[0]].values.ravel()
	yy_pred = df[quantitativeColumns[1]].values.ravel()
	confusion = confusion_matrix(yy_test, yy_pred)
	data['confusion'].append(confusion)
	plt.matshow(confusion)
	plt.title('Confusion matrix')
	plt.gray()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

	save_bytes_image(image_list)
	plt.clf()

	invert_colors = np.ones(confusion.shape) * confusion.max()
	plt.matshow(invert_colors - confusion)
	plt.title('Confusion matrix')
	plt.gray()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

	save_bytes_image(image_list)
	
	res = {
		'output': image_list,
		'result': pd.DataFrame(data).head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(pd.DataFrame(data).head(10))
	return res

res = demo_mat_show(self.current_df, self.intermediate_df, description, method)
