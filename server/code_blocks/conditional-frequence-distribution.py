import pandas as pd
import numpy as np
import nltk
nltk.download('brown')

def conditional_frequence_distribution(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset.select_dtypes(include='object')
	
	if df.empty == True: 
		res = {
			'output': "Dataframe has no object values", 
			'result' : "Dataframe has no object values",
			'description' : "Dataframe has no object values",
			'type' : "error"
		}
		print (res['output'])
		return res

	# if len(intermediate_df) != 0:
	# 	df = intermediate_df[-1]
	# else:
	# 	df = loaded_dataset

	res_df = calcConditionalFreqDist(df)

	res = {
		'output': df.head(10).to_json(orient='table'),
		'result': res_df.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(df.head(10))
	print (res['output'])
	return res

def calcConditionalFreqDist(df):

    # words = ['can', 'could', 'may', 'might', 'must', 'will']
    words = (df.select_dtypes(include='object').values).ravel()
    genres = ['adventure', 'romance', 'science_fiction']
    
    cfdist = nltk.ConditionalFreqDist(
                (genre, word)
                for genre in genres
                for word in nltk.corpus.brown.words(categories=genre)
                if word in words)

    data = {'conditionalDist': cfdist}
    return pd.DataFrame(data)


# df = pd.DataFrame(np.random.uniform(low=False, high=True, size=(29, 10)), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'])

df = pd.DataFrame({'a': [0, 0] * 5, 'b': [0, 0] * 5,  'c': [1, 1] * 5})
# print (df)


res = conditional_frequence_distribution(self.current_df, self.intermediate_df, description, method)
