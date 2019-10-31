import pandas as pd
import os
import io
import json
import base64
import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

import seaborn

class Analysis():
    def __init__(self):
        self.intermediate_df = []
        self.loaded_dataset = pd.DataFrame()
        self.current_df = pd.DataFrame()
        self.dataset = ""
        self.intermediate_selected = False

    def dataset_changed(self, dataset):
            self.dataset = dataset
            self.loaded_dataset = load_dataset(dataset)
            self.intermediate_df = []
            self.current_df = []    

    def delete_intermediate_df(self, index):
        self.intermediate_df = self.intermediate_df[:int(index)]
        print('length is ' + str(len(self.intermediate_df)))

    def export_intermediate_df(self, index):
        # NOTE: this may not work for some cells, e.g., shuffle-split cell
        idx = int(index)
        if idx < len(self.intermediate_df):
            return self.intermediate_df[idx]
        else:
            return pd.DataFrame()

    def execute_analysis(self, method, dataset, state=None):
        import numpy as np
        # once the dataset changes, initialize new instance variables
        if self.dataset != dataset:
            self.dataset_changed(dataset)

        if len(self.intermediate_df) == 0:
            self.loaded_dataset = load_dataset(dataset)
            self.current_df = self.loaded_dataset

        description = ''
        #load in json file
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            if method in json_data[i]["user-data"]["method"]:
                description = json_data[i]['description']
                break
            
        code_string = ""

        for i in range(0, len(json_data)):
            if method in json_data[i]["user-data"]["method"]:
                code_string = ''.join(map(str, json_data[i]['code']))
        
       
        print ("Current Dataframe: \n")
        print (self.current_df)
        exec(code_string)
                
        print ("Response from block execution: \n")
        return res
        
def load_dataset(filename):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/../../data/', filename + ".csv")
    df = pd.read_csv(data_path)
    return df


def save_bytes_image(image_list):
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    image_list.append(base64.b64encode(bytes_image.getvalue()))
    bytes_image.seek(0)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

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


from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true,y_predict)
    
    # Return the score
    return score