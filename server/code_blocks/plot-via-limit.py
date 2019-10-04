import pandas as pd
import numpy  as np
import io
import base64
import matplotlib.pyplot as plt



def plot_via_limit(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset.select_dtypes(include='number')

	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1].select_dtypes(include='number')
	# else:
	# 	df = loaded_dataset.select_dtypes(include='number')

	if df.empty == True or len(df.columns) < 2: 
		res = {
			'output': "Dataframe has no numeric values", 
			'result': "Dataframe has no numeric values", 
			'description' : "Dataframe has no numeric values",
			'type' : "error"
		}
		print (res['output'])
		return res

	image_list = []
	p_steps = 100
	v_steps = 100
	P = [item for sublist in df[df.columns[[0]]].values.tolist() for item in sublist]
	V = [item for sublist in df[df.columns[[1]]].values.tolist() for item in sublist]
	x = np.arange(-np.pi, np.pi, 2*np.pi/p_steps)
	y = []
	for i in range(len(x)): y.append([])
	for p, v in zip(P, V):
		i = int((p+np.pi)/(2*np.pi/p_steps))
		y[i].append(v)
	
	means = [ np.mean(np.array(vs)) for vs in y ]
	stds = [ np.std(np.array(vs)) for vs in y ]
	plt.plot(x, means)
	plt.plot([-2, -2], [-0.99, 0.99], 'k:', lw=1)
	plt.plot([2, 2], [-0.99, 0.99], 'k:', lw=1)
	plt.xlim([-np.pi, np.pi])
	plt.ylim([-1,1])
	plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
	plt.xlabel(r"Phase [rad]")
	plt.ylabel(r"V mean")
	save_bytes_image(image_list)

	res = {
		'output': df.head(10).to_json(orient='table'),
		'result': image_list,
		'description' : description,
		'type': method
	}
	
	intermediate_df.append( df.head(10))
	print(res['output'])
	return res


def save_bytes_image(image_list):
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	image_list.append(base64.b64encode(bytes_image.getvalue()))
	bytes_image.seek(0)


# df = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 3)), columns=['a', 'b', 'c'])

# df = pd.DataFrame({'a': [0, 1] * 5, 'b': [1, 10] * 5,  'c': [12, 13] * 5})

res = plot_via_limit(self.current_df, self.intermediate_df, description, method)