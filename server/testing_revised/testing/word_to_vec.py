def word_to_vec(loaded_dataset, intermediate_df, description, method):
	df= loaded_dataset
	
	#this function might be throwing errors - still needs to be looked at
	def calcWordVec(df):
		texts = df.select_dtypes(include='object')

		MAX_NB_WORDS = 5000
		EMBEDDING_DIM = 100
		
		tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
		sequences = tokenizer.texts_to_sequences(texts)
		word_index = tokenizer.word_index

		nb_words = min(MAX_NB_WORDS, len(word_index))+1

		embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
		for word, i in word_index.items():
				if word in word2vec.vocab:
					embedding_matrix[i] = word2vec.word_vec(word)
		print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

		data = {"wordvec":[]}
		data['wordvec'].append(np.sum(np.sum(embedding_matrix, axis=1) == 0))
		return pd.DataFrame(data)

	try:
		res_df = calcWordVec(df)
	except Exception as e: 
		res = {
			'output': str(e), 
			'result': str(e), 
			'description' : str(e),
			'type' : "error"
		}
		return res
	
		res = {
		'output': res_df.head(10).to_json(orient='table'),
		'result': res_df.head(10).to_json(orient='table'),
		'description' : description,
		'type': method
	}
	intermediate_df.append(res_df.head(10))
	return res

