def fit_decision_tree(loaded_dataset, intermediate_df, description, method):
	df = loaded_dataset

	from sklearn.tree import DecisionTreeRegressor
	from sklearn.metrics import make_scorer
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import ShuffleSplit
	from sklearn.metrics import r2_score

	def performance_metric(y_true, y_predict):
		""" Calculates and returns the performance score between 
			true and predicted values based on the metric chosen. """
		
		# TODO: Calculate the performance score between 'y_true' and 'y_predict'
		score = r2_score(y_true,y_predict)
		
		# Return the score
		return score
		
	def fit_model(X, y):
		""" Performs grid search over the 'max_depth' parameter for a 
			decision tree regressor trained on the input data [X, y]. """
		
		# Create cross-validation sets from the training data
		cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

		# TODO: Create a decision tree regressor object
		regressor = DecisionTreeRegressor()

		# TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
		params = {'max_depth': range(1,11)}

		# TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
		scoring_fnc = make_scorer(performance_metric)

		# TODO: Create the grid search object
		grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)

		# Fit the grid search object to the data to compute the optimal model
		grid = grid.fit(X, y)

		# Return the optimal model after fitting the data
		return grid.best_estimator_
		
	X_train, X_test, y_train, y_test = df


	try:
		reg = fit_model(X_train, y_train)
	except Exception as e:
		res = {
			'output': str(e),
			'result': str(e),
			'description' : str(e),
			'type': 'error'
		}
		return res
	
	res = {
		'output' : y_train.head(10).round(3).to_json(orient='table'),
		'result' : "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']),
		'description' : description,
		'type' : method
	}
	intermediate_df.append(y_train.head(10).round(3))
	return res

res = fit_decision_tree(self.current_df, self.intermediate_df, description, method)