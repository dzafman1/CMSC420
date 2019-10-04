import pandas as pd
import os
import io
import json
import base64
import itertools

import matplotlib
matplotlib.use('TkAgg')
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

    def dataset_changed(self, dataset):
            self.dataset = dataset
            self.loaded_dataset = load_dataset(dataset)
            self.intermediate_df = []
            self.current_df = []    

    def delete_intermediate_df(self, index):
        self.intermediate_df = self.intermediate_df[:int(index)]
        print 'length is ' + str(len(self.intermediate_df))

    def export_intermediate_df(self, index):
        # NOTE: this may not work for some cells, e.g., shuffle-split cell
        idx = int(index)
        if idx < len(self.intermediate_df):
            return self.intermediate_df[idx]
        else:
            return pd.DataFrame()

    def execute_analysis(self, method, dataset, state=None):
        # once the dataset changes, initialize new instance variables
        if self.dataset != dataset:
            self.dataset_changed(dataset)

        if len(self.intermediate_df) == 0:
            self.loaded_dataset = load_dataset(dataset)
            self.current_df = self.loaded_dataset
        else:
            self.current_df = self.intermediate_df[-1]

        description = ''
        #load in json file
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            if method in json_data[i]["user-data"]["method"]:
                description = json_data[i]['description']
                break

        # if method == 'category-count':
        #     category_df = self.current_df.select_dtypes(include='object')
        #     image_list = []
        #     for col in category_df:
        #         if category_df[col].value_counts().count() <= 20:
        #             seaborn.catplot(x=col, data=category_df, alpha=0.7, kind='count')
        #             save_bytes_image(image_list)
        #     res = {
        #         'output' : category_df.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res


        # elif method == 'distribution-numerical':
        #     image_list = []
        #     numerical_df = self.current_df.select_dtypes(include='number')
        #     count = 0

        #     for col in numerical_df:

        #         fig, ax = plt.subplots()
        #         ax.hist(numerical_df[col])
        #         plt.xlabel(col)
        #         plt.ylabel("Dist")
        #         plt.title('Histogram of ' + col)

        #         save_bytes_image(image_list)
        #         count += 1
        #         if count >= 5:
        #             break

        #     res = {
        #         'output' : numerical_df.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res

        # # TODO: not sure if we still need this
        # elif method == 'feature-selection':
        #     column_list = list(self.loaded_dataset)
        #     res = {
        #         'result' : column_list,
        #         'type' : method
        #     }
        #     return res

        # elif method == 'scatterplot-regression':
        #     # print state
        #     # loaded_state = json.loads(state)
        #     # print(loaded_state)

        #     # features = loaded_state['features']
        #     # target = loaded_state['target']
        #     numerical_df = self.current_df.select_dtypes(include='number')
        #     image_list = []
        #     count = 0
        #     for col1, col2 in itertools.combinations(numerical_df, 2):
        #         plt.clf()
        #         seaborn.regplot(self.loaded_dataset[col1], self.loaded_dataset[col2])
        #         save_bytes_image(image_list)
        #         count += 1
        #         if count >= 5:
        #             break

        #     res = {
        #         'output' : numerical_df.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res


        # elif method == 'stack-facetgrid':
        #     loaded_dataset = self.current_df

        #     numerical_cols = loaded_dataset.select_dtypes(include='number').columns
        #     category_cols = loaded_dataset.select_dtypes(include='object').columns
           
        #     image_list = []
        #     for cat_var in category_cols:
        #         if loaded_dataset[cat_var].value_counts().count() <= 5:
        #             for num_var in numerical_cols:
        #                 plt.clf()
        #                 fig = seaborn.FacetGrid(loaded_dataset,hue=cat_var)
        #                 fig.map(seaborn.kdeplot,num_var,shade=True)
        #                 oldest = loaded_dataset[num_var].max()
        #                 fig.set(xlim=(0, oldest))
        #                 fig.add_legend()
        #                 save_bytes_image(image_list) 

        #                 if len(image_list) >= 5:
        #                     break
                        
        #     res = {
        #         'output' : loaded_dataset.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res


        # elif method == 'drop-NaN-columns':
        #     df = self.current_df
        #     dropped_columns = []
        #     df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
        #     for c in df.columns:
        #         if c not in df2.columns:
        #             dropped_columns.append(c)

        #     res = {
        #         'output' : df2.describe().to_json(orient='table'),
        #         'result' : df2.describe().to_json(orient='table'),
        #         'type' : method
        #     }

        #     self.intermediate_df.append(df2)

        #     return res

        
        # elif method == 'correlation-heatmap':
        #     corr = self.current_df.select_dtypes(include='number').corr()

        #     image_list = []
        #     plt.clf()
        #     seaborn.heatmap(corr, 
        #     cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
        #     annot=True, annot_kws={"size": 8}, square=True)

        #     save_bytes_image(image_list)
        #     res = {
        #         'output' : corr.to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)

        #     return res

        
        # elif method == 'category-boxplot':
        #     loaded_dataset = self.current_df

        #     numerical_cols = loaded_dataset.select_dtypes(include='number').columns
        #     category_cols = loaded_dataset.select_dtypes(include='object').columns

        #     image_list = []
        #     for cat_var in category_cols:
        #         if loaded_dataset[cat_var].value_counts().count() <= 5:
        #             for num_var in numerical_cols:
        #                 plt.clf()
        #                 ax = seaborn.boxplot(x=cat_var, y=num_var, data=loaded_dataset)
        #                 plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
        #                 plt.xticks(rotation=45)

        #                 save_bytes_image(image_list)

        #                 if len(image_list) >= 5:
        #                     break
      
        #         if len(image_list) >= 5:
        #             break

        #     res = {
        #         'output' : loaded_dataset.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)

        #     return res

        
        # elif method == 'numerical-boxplot':
        #     loaded_dataset = self.current_df

        #     numerical_cols = loaded_dataset.select_dtypes(include='number').columns

        #     image_list = []
        #     for num_var in numerical_cols:
        #         plt.clf()
        #         ax = seaborn.boxplot(y=num_var, data=loaded_dataset)
        #         plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
        #         plt.xticks(rotation=45)

        #         save_bytes_image(image_list)

        #         if len(image_list) >= 5:
        #             break

        #     res = {
        #         'output' : loaded_dataset.select_dtypes(include='number').head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res


        # elif method == 'distribution-two-categories':
        #     loaded_dataset = self.current_df
        #     image_list = []
        #     clean_dataset = pd.DataFrame()
        #     for col in loaded_dataset.columns:
        #         if loaded_dataset[col].dtype == 'object' or loaded_dataset[col].value_counts().count() <= 20:
        #             clean_dataset[col] = loaded_dataset[col].astype('category')
        #     category_df = clean_dataset.select_dtypes(include='category')
        #     count = 0

        #     for col1, col2 in itertools.combinations(category_df.columns, 2):
        #         if category_df[col1].value_counts().count() <= 10 and \
        #             category_df[col2].value_counts().count() <= 5:
        #             seaborn.catplot(col1, data=category_df, hue=col2, kind='count')

        #             save_bytes_image(image_list)
        #             count += 1

        #             if count >= 5:
        #                 break

        #     res = {
        #         'output' : category_df.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res


        # elif method == 'distribution-quantitative-category':
        #     loaded_dataset = self.current_df
        #     image_list = []
        #     clean_dataset = pd.DataFrame()
        #     for col in loaded_dataset.columns:
        #         if loaded_dataset[col].dtype == 'object' or loaded_dataset[col].value_counts().count() <= 20:
        #             clean_dataset[col] = loaded_dataset[col].astype('category')
        #         else:
        #             clean_dataset[col] = loaded_dataset[col]

        #     numerical_cols = clean_dataset.select_dtypes(include='number').columns
        #     category_cols = clean_dataset.select_dtypes(include='category').columns
        #     count = 0

        #     for col1, col2 in zip(numerical_cols, category_cols):
        #         if clean_dataset[col2].value_counts().count() <= 10:
        #             seaborn.catplot(col2, col1, data=clean_dataset)

        #             save_bytes_image(image_list)
        #             count += 1

        #             if count >= 5:
        #                 break

        #     res = {
        #         'output' : clean_dataset.head(10).to_json(orient='table'),
        #         'result' : image_list,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        #     return res
        
        # elif method == 'top5categories':
        #     loaded_dataset = self.current_df
        #     category_df = loaded_dataset.select_dtypes(include='object')

        #     for col in category_df:
        #         samples = loaded_dataset.value_counts().head(5)
        #         res = {
        #             'output' : samples.to_json(orient='table'),
        #             'result' : samples.to_json(orient='table'),
        #             'type' : method
        #         }
        #         break
            
        #     self.intermediate_df.append(self.current_df)

        #     return res

        if method == 'shuffle-split':
            loaded_dataset = self.current_df
            numerical_df = loaded_dataset.select_dtypes(include='number')
            features = numerical_df[numerical_df.columns[0:3]]
            predicted_variables = numerical_df[numerical_df.columns[-1]]

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(features, predicted_variables, test_size=0.2,random_state=100)

            new_df = (X_train, X_test, y_train, y_test)
            self.intermediate_df.append(new_df)
            res = {
                'output' : X_train.head(10).round(3).to_json(orient="table"),
                'result' : "split into training and testing set",
                'description' : description,
                'type' : method
            }

        elif method == 'fit-decision-tree':
            X_train, X_test, y_train, y_test = self.current_df

            reg = fit_model(X_train, y_train)
            self.intermediate_df.append(reg)

            res = {
                'output' : y_train.head(10).round(3).to_json(orient='table'),
                'result' : "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']),
                'description' : description,
                'type' : method
            }

        elif method == 'predict-test':
            reg = self.intermediate_df[-1]
            X_train, X_test, y_train, y_test = self.intermediate_df[-2]

            y_predict = reg.predict(X_test)

            new_df = pd.DataFrame()
            new_df['predicted'] = y_predict
            new_df['actual'] = list(y_test)

            res = {
                'output' : new_df.head(10).round(3).to_json(orient='table'),
                'result' : "predicted vs. actual test done, see Output Data Frame Tab.",
                'description' : description,
                'type' : method
            }

        elif method == 'drop-NaN-rows': 
            new_df = self.current_df.dropna()
            res = {
                'output' : new_df.head(10).round(3).to_json(orient='table'),
                'result' : new_df.describe().round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(new_df)
        
        elif method == 'mean':
            new_df = pd.DataFrame(self.current_df.mean(), columns=['mean'])
            print(new_df)
            res = {
                'output' : self.current_df.head(10).round(3).to_json(orient='table'),
                'result' : new_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(self.current_df)

        elif method == 'variance':
            new_df = self.current_df.var()
            res = {
                'output' : self.current_df.head(10).round(3).to_json(orient='table'),
                'result' : new_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(self.current_df)
        
        elif method == 'ranksum-test':
            current_df = self.current_df
            numerical_df = current_df.select_dtypes(include='number')
            # create a dataframe with columns and rows have the same name to store test values
            res_df = pd.DataFrame(columns=(numerical_df.columns), index=(numerical_df.columns))

            for col1, col2 in itertools.combinations(numerical_df, 2):
                z_stat, p_val = stats.ranksums(numerical_df[col1], numerical_df[col2])
                res_df[col1][col2] = p_val
                res_df[col2][col1] = p_val

            res = {
                'output' : self.current_df.head(10).round(3).to_json(orient='table'),
                'result' : res_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(self.current_df)

        elif method == 'ANOVA-Variance-Analysis':
            current_df = self.current_df
            category_cols = current_df.select_dtypes(include='object').columns
            numerical_cols = current_df.select_dtypes(include='number').columns
            res_df = pd.DataFrame(columns=category_cols, index=numerical_cols)

            for num_col in numerical_cols:
                for cat_col in category_cols:
                    if current_df[cat_col].value_counts().count() <= 10:
                        groups = current_df.groupby(cat_col).groups.keys()
                        print groups
                        print current_df[current_df[cat_col] == groups[0]][num_col]
                        # ANOVA for only the first three groups
                        if len(groups) >= 3:
                            f_val, p_val = stats.f_oneway(current_df[current_df[cat_col] == groups[0]][num_col], current_df[current_df[cat_col] == groups[1]][num_col], current_df[current_df[cat_col] == groups[2]][num_col])
                            res_df[cat_col][num_col] = p_val

            res = {
                'output' : self.current_df.head(10).round(3).to_json(orient='table'),
                'result' : res_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(self.current_df)

        elif method == 'bootstrap':
            current_df = self.current_df.select_dtypes('number')
            from sklearn.utils import resample
            
            mean = []
            statistics = pd.DataFrame()
            # prepare bootstrap sample
            for i in range(0, 1000):
                boot = resample(current_df, replace=True, n_samples=int(0.5 * len(current_df.index)))
                mean.append(boot.mean().to_dict())

            for key in mean[0]:
                curr_list = [item[key] for item in mean]
                # confidence intervals
                alpha = 0.95
                p = (1.0-alpha) * 100
                lower = np.percentile(curr_list, p)
                p = alpha * 100
                upper = np.percentile(curr_list, p)
                statistics[key] = [str(round(lower, 3)) + '-' + str(round(upper, 3))]
            
            res = {
                'output' : self.current_df.head(10).round(3).to_json(orient='table'),
                'result' : statistics.to_json(orient='table'),
                'description' : description,
                'type' : method
            }

            self.intermediate_df.append(self.current_df)
            
        # automated blocks start from here
        elif method == 'matrix-norm':
            import numpy.linalg as LA
            from pandas.api.types import is_numeric_dtype
            loaded_dataset = self.current_df
            nCols = [c for c in list(loaded_dataset) if is_numeric_dtype(loaded_dataset[c])]
            data = {"columnName":[],"NumpyLinalgNorm":[]}

            for nc in nCols:
                data["columnName"].append(nc)
                data["NumpyLinalgNorm"].append(LA.norm(loaded_dataset[[nc]].values))
            
            res = {
                'output': pd.DataFrame(data).to_json(orient='table'),
                'result': pd.DataFrame(data).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            self.intermediate_df.append(self.current_df)

        elif method == 'test-linear-regression':
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            from pandas.api.types import is_numeric_dtype

            loaded_dataset = self.current_df
            linear_regression = LinearRegression()
            quantitativeColumns = [c for c in list(loaded_dataset) if is_numeric_dtype(loaded_dataset[c])]
            data = loaded_dataset[quantitativeColumns[:-1]]
            target = loaded_dataset[[quantitativeColumns[-1]]].values

            res = {
                'output': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
                'result': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            self.intermediate_df.append(self.current_df)

        elif method == "eval-model-predictions":

            df = self.current_df.select_dtypes(include='number')

            predictions = df.iloc[:,-1].values # last column
            labels = df.iloc[:,-2].values # second to last column

            res = {
                'output': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
                'result': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            self.intermediate_df.append(self.current_df)
        
        elif method == "test-randforest-classifier":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from pandas.api.types import is_numeric_dtype

            df = self.current_df.select_dtypes(include='number')

            forest = RandomForestClassifier(n_estimators=100,
                n_jobs=-1,random_state=17)
            quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
            X_train = df[quantitativeColumns[:-1]]
            y_train = df[[quantitativeColumns[-1]]].values.ravel()

            res = {
                'output': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
                'result': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            self.intermediate_df.append(self.current_df)

        elif method == "compute-percentiles-range":
            from pandas.api.types import is_numeric_dtype

            lowerPercentile = 5
            upperPercentile = 95
            df = self.current_df
            
            quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
            data = {"Percentile"+str(lowerPercentile):[],"Percentile"+str(upperPercentile):[],"columnName":quantitativeColumns}
            for c in quantitativeColumns:
                data["Percentile"+str(lowerPercentile)].append(np.percentile(df[[c]],lowerPercentile))
                data["Percentile"+str(upperPercentile)].append(np.percentile(df[[c]],upperPercentile))
            res = {
                'output': pd.DataFrame(data).to_json(orient='table'),
                'result': pd.DataFrame(data).to_json(orient='table'),
                'description' : description,
                'type': method
            }

        else:
            
            code_string = ""

            for i in range(0, len(json_data)):
                if method in json_data[i]["user-data"]["method"]:
                    code_string = ''.join(map(str, json_data[i]['code']))
            
            exec(code_string)
            # NOTE: this may not work for some analysis since they are not in dict yet
            res['code'] = code_string
        
        print (len(self.intermediate_df))

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