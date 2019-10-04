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

        # if method == 'shuffle-split':
        #     loaded_dataset = self.current_df
        #     numerical_df = loaded_dataset.select_dtypes(include='number')
        #     features = numerical_df[numerical_df.columns[0:3]]
        #     predicted_variables = numerical_df[numerical_df.columns[-1]]

        #     from sklearn.model_selection import train_test_split
        #     X_train, X_test, y_train, y_test = train_test_split(features, predicted_variables, test_size=0.2,random_state=100)

        #     new_df = (X_train, X_test, y_train, y_test)
        #     self.intermediate_df.append(new_df)
        #     res = {
        #         'output' : X_train.head(10).round(3).to_json(orient="table"),
        #         'result' : "split into training and testing set",
        #         'description' : description,
        #         'type' : method
        #     }

        # elif method == 'fit-decision-tree':
        #     X_train, X_test, y_train, y_test = self.current_df

        #     reg = fit_model(X_train, y_train)
        #     self.intermediate_df.append(reg)

        #     res = {
        #         'output' : y_train.head(10).round(3).to_json(orient='table'),
        #         'result' : "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']),
        #         'description' : description,
        #         'type' : method
        #     }

        # elif method == 'predict-test':
        #     reg = self.intermediate_df[-1]
        #     X_train, X_test, y_train, y_test = self.intermediate_df[-2]

        #     y_predict = reg.predict(X_test)

        #     new_df = pd.DataFrame()
        #     new_df['predicted'] = y_predict
        #     new_df['actual'] = list(y_test)

        #     res = {
        #         'output' : new_df.head(10).round(3).to_json(orient='table'),
        #         'result' : "predicted vs. actual test done, see Output Data Frame Tab.",
        #         'description' : description,
        #         'type' : method
        #     }

        # if method == 'drop-NaN-rows': 
        #     new_df = self.current_df.dropna()
        #     res = {
        #         'output' : new_df.head(10).round(3).to_json(orient='table'),
        #         'result' : new_df.describe().round(3).to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(new_df)
        
        # elif method == 'mean':
        #     new_df = pd.DataFrame(self.current_df.mean(), columns=['mean'])
        #     print(new_df)
        #     res = {
        #         'output' : self.current_df.head(10).round(3).to_json(orient='table'),
        #         'result' : new_df.round(3).to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # elif method == 'variance':
        #     new_df = self.current_df.var()
        #     res = {
        #         'output' : self.current_df.head(10).round(3).to_json(orient='table'),
        #         'result' : new_df.round(3).to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
        
        # elif method == 'ranksum-test':
        #     current_df = self.current_df
        #     numerical_df = current_df.select_dtypes(include='number')
        #     # create a dataframe with columns and rows have the same name to store test values
        #     res_df = pd.DataFrame(columns=(numerical_df.columns), index=(numerical_df.columns))

        #     for col1, col2 in itertools.combinations(numerical_df, 2):
        #         z_stat, p_val = stats.ranksums(numerical_df[col1], numerical_df[col2])
        #         res_df[col1][col2] = p_val
        #         res_df[col2][col1] = p_val

        #     res = {
        #         'output' : self.current_df.head(10).round(3).to_json(orient='table'),
        #         'result' : res_df.round(3).to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # elif method == 'ANOVA-Variance-Analysis':
        #     current_df = self.current_df
        #     category_cols = current_df.select_dtypes(include='object').columns
        #     numerical_cols = current_df.select_dtypes(include='number').columns
        #     res_df = pd.DataFrame(columns=category_cols, index=numerical_cols)

        #     for num_col in numerical_cols:
        #         for cat_col in category_cols:
        #             if current_df[cat_col].value_counts().count() <= 10:
        #                 groups = current_df.groupby(cat_col).groups.keys()
        #                 print groups
        #                 print current_df[current_df[cat_col] == groups[0]][num_col]
        #                 # ANOVA for only the first three groups
        #                 if len(groups) >= 3:
        #                     f_val, p_val = stats.f_oneway(current_df[current_df[cat_col] == groups[0]][num_col], current_df[current_df[cat_col] == groups[1]][num_col], current_df[current_df[cat_col] == groups[2]][num_col])
        #                     res_df[cat_col][num_col] = p_val

        #     res = {
        #         'output' : self.current_df.head(10).round(3).to_json(orient='table'),
        #         'result' : res_df.round(3).to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # elif method == 'bootstrap':
        #     current_df = self.current_df.select_dtypes('number')
        #     from sklearn.utils import resample
            
        #     mean = []
        #     statistics = pd.DataFrame()
        #     # prepare bootstrap sample
        #     for i in range(0, 1000):
        #         boot = resample(current_df, replace=True, n_samples=int(0.5 * len(current_df.index)))
        #         mean.append(boot.mean().to_dict())

        #     for key in mean[0]:
        #         curr_list = [item[key] for item in mean]
        #         # confidence intervals
        #         alpha = 0.95
        #         p = (1.0-alpha) * 100
        #         lower = np.percentile(curr_list, p)
        #         p = alpha * 100
        #         upper = np.percentile(curr_list, p)
        #         statistics[key] = [str(round(lower, 3)) + '-' + str(round(upper, 3))]
            
        #     res = {
        #         'output' : self.current_df.head(10).round(3).to_json(orient='table'),
        #         'result' : statistics.to_json(orient='table'),
        #         'description' : description,
        #         'type' : method
        #     }

        #     self.intermediate_df.append(self.current_df)
            
        # automated blocks start from here
        # elif method == 'matrix-norm':
        #     import numpy.linalg as LA
        #     from pandas.api.types import is_numeric_dtype
        #     loaded_dataset = self.current_df
        #     nCols = [c for c in list(loaded_dataset) if is_numeric_dtype(loaded_dataset[c])]
        #     data = {"columnName":[],"NumpyLinalgNorm":[]}

        #     for nc in nCols:
        #         data["columnName"].append(nc)
        #         data["NumpyLinalgNorm"].append(LA.norm(loaded_dataset[[nc]].values))
            
        #     res = {
        #         'output': pd.DataFrame(data).to_json(orient='table'),
        #         'result': pd.DataFrame(data).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # elif method == 'test-linear-regression':
        #     from sklearn.linear_model import LinearRegression
        #     from sklearn.model_selection import cross_val_score
        #     from pandas.api.types import is_numeric_dtype

        #     loaded_dataset = self.current_df
        #     linear_regression = LinearRegression()
        #     quantitativeColumns = [c for c in list(loaded_dataset) if is_numeric_dtype(loaded_dataset[c])]
        #     data = loaded_dataset[quantitativeColumns[:-1]]
        #     target = loaded_dataset[[quantitativeColumns[-1]]].values

        #     res = {
        #         'output': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
        #         'result': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # elif method == "eval-model-predictions":

        #     df = self.current_df.select_dtypes(include='number')

        #     predictions = df.iloc[:,-1].values # last column
        #     labels = df.iloc[:,-2].values # second to last column

        #     res = {
        #         'output': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
        #         'result': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }

        #     self.intermediate_df.append(self.current_df)
        
        # elif method == "random-forest-classifier":
        #     from sklearn.ensemble import RandomForestClassifier
        #     from sklearn.model_selection import cross_val_score
        #     from pandas.api.types import is_numeric_dtype

        #     df = self.current_df.select_dtypes(include='number')

        #     forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=17)
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     X_train = df[quantitativeColumns[:-1]]
        #     y_train = df[[quantitativeColumns[-1]]].values.ravel()

        #     res = {
        #         'output': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
        #         'result': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }

        #     self.intermediate_df.append(self.current_df)

        # if method == "compute-percentiles-range":
        #     from pandas.api.types import is_numeric_dtype

        #     lowerPercentile = 5
        #     upperPercentile = 95
        #     df = self.current_df
            
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     data = {"Percentile"+str(lowerPercentile):[],"Percentile"+str(upperPercentile):[],"columnName":quantitativeColumns}
        #     for c in quantitativeColumns:
        #         data["Percentile"+str(lowerPercentile)].append(np.percentile(df[[c]],lowerPercentile))
        #         data["Percentile"+str(upperPercentile)].append(np.percentile(df[[c]],upperPercentile))
        #     res = {
        #         'output': pd.DataFrame(data).to_json(orient='table'),
        #         'result': pd.DataFrame(data).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "compute-covariance-matrix":
        #     data = {"CovMeanDot":[]}
    
        #     df_matrix = self.current_df._get_numeric_data().values

        #     covariance = np.cov(df_matrix)
        #     mean = np.mean(df_matrix, axis=0)
        #     inv = np.linalg.inv(covariance)
        #     dot = np.dot(np.dot(mean, inv), mean)

        #     data["CovMeanDot"].append(dot)

        #     res = {
        #         'output': pd.DataFrame(data).to_json(orient='table'),
        #         'result': pd.DataFrame(data).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "compute-sparse-linearsystem":
        #     import scipy.sparse.linalg
        #     from pandas.api.types import is_numeric_dtype
        #     from scipy import sparse
        #     df = self.current_df

        #     nCols = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     A = df[nCols[:-1]].values
        #     if A.shape[0] != A.shape[1]:
        #         return None
        #     x = df[[nCols[-1]]].values

        #     b = A.dot(x)
        #     sA = sparse.csr_matrix(A)
        #     x = scipy.sparse.linalg.spsolve(sA, b)
        #     print x
        #     res = {
        #         'output': pd.DataFrame(x).to_json(orient='table'),
        #         'result': pd.DataFrame(x).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     print res
        #     self.intermediate_df.append(self.current_df)

        # elif method == "decision-tree-classifier":
        #     df = self.current_df

        #     from sklearn.tree import DecisionTreeClassifier
        #     from sklearn.model_selection import train_test_split
        #     from pandas.api.types import is_numeric_dtype
        #     from sklearn.metrics import accuracy_score
            
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     X = df[quantitativeColumns[:-1]]
        #     y = df[[quantitativeColumns[-1]]].values.ravel()

        #     classifier = DecisionTreeClassifier()
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        #     classifier.fit(X_train, y_train)
        #     prediction = classifier.predict(X_test)

        #     data = {'accuracyScore': []}
        #     data['accuracyScore'].append(accuracy_score(y_test, prediction, normalize=False))
  
        #     res = {
        #         'output': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'result': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "logistic-regression":
        #     df = self.current_df
        #     image_list = []

        #     from sklearn.linear_model import LogisticRegression
        #     from sklearn.model_selection import train_test_split
            
        #     alt_df = df.select_dtypes(include='number')

        #     h = alt_df.shape[0]
        #     alt_data = alt_df[alt_df.columns[[0, 2]]]

        #     X = alt_data._get_numeric_data().values
        #     y = np.arange(h)

        #     logit = LogisticRegression(random_state=0, n_jobs=-1)
        #     logit.fit(X, y)
        #     theta = logit.predict(X)

        #     X_flat = np.array([item for sublist in X.tolist() for item in sublist])

        #     max = np.max(X_flat)
        #     min = np.min(X_flat)

        #     a = np.linspace(min, max, len(theta))
        #     z = (theta[0] + theta[1]*a)/theta[2]

        #     plt.plot(y, z)
        #     plt.scatter(df[['a']], df[['b']], c='green', label='data')
        #     plt.legend()
        #     save_bytes_image(image_list)
        #     data = {"logReg":[]}
        #     data["logReg"].append(z)

        #     res = {
        #         'output': image_list,
        #         'result': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "demo-hstack":
        #     df = self.current_df

        #     from pandas.api.types import is_numeric_dtype
        #     import numpy as np

        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     x = df[quantitativeColumns[0]].values.ravel()
        #     y = df[quantitativeColumns[1]].values.ravel()

        #     x1 = 1 / x
        #     y1 = 1 / y

        #     # x1, y1 = 1 / np.random.uniform(-1000, 100, size=(2, 10000))
        #     x2, y2 = np.dot(np.random.uniform(size=(2, 2)), np.random.normal(size=(2, len(x))))
        #     u = np.hstack([x1, x2])
        #     v = np.hstack([y1, y2])


        #     res = {
        #         'output': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
        #         'result': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "extra-trees-classifier":
        #     df = self.current_df

        #     from sklearn.ensemble import ExtraTreesClassifier
        #     from sklearn.model_selection import cross_val_score
        #     from sklearn.model_selection import train_test_split
        #     from pandas.api.types import is_numeric_dtype
        #     from sklearn.metrics import accuracy_score
        #     # from sklearn.metrics import classification_report
        #     # from sklearn.metrics import confusion_matrix

        #     ETC = ExtraTreesClassifier()
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     X = df[quantitativeColumns[:-1]]
        #     y = df[[quantitativeColumns[-1]]].values.ravel()
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
        #     ETC = ExtraTreesClassifier()
        #     ETC.fit(X_train, y_train)

        #     prediction = ETC.predict(X_test)

        #     # print("Accuracy Score", accuracy_score(y_test, prediction))
        #     # print(classification_report(y_test, prediction,target_names=['target_1', 'target_2']))
        #     # print(confusion_matrix(y_test, prediction))
        #     # print(accuracy_score(y_test,prediction))
        #     # print("Time to run", time.clock() - start_time, "seconds")

        #     #verify with Cross Validation
        #     scores = cross_val_score(ETC,X_train,y_train,cv=2)
        #     # print("\n")
        #     # print("Cross Validation Score",scores)
        #     # print("Average of Cross Validation.", scores.mean())

        #     res = {
        #         'output': pd.DataFrame(scores).to_json(orient='table'),
        #         'result': pd.DataFrame(scores).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "unique-column-values":
        #     test = {}
        #     df = self.current_df
        #     alt_df = df.select_dtypes(include='number')

        #     for column in alt_df:
        #         test[column] = alt_df.dropna().unique()

        #     res = {
        #         'output': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
        #         'result': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "demo-log-space":
        #     df = self.current_df
            
        #     from pandas.api.types import is_numeric_dtype
        #     from sklearn.model_selection import train_test_split
        #     import numpy as np

        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     X = df[quantitativeColumns[:-1]]
        #     y = df[[quantitativeColumns[-1]]].values.ravel()
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        #     data = {'svc__C': np.logspace(-3, 2, 6), 'svc__gamma': np.logspace(-3, 2, 6) / X_train.shape[0]}

        #     res = {
        #         'output': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'result': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "probability-density-plot":
        #     df = self.current_df
        #     image_list = []
            
        #     from scipy.stats import chi2
        #     from pandas.api.types import is_numeric_dtype

        #     data = {'rv':[], 'pdf':[]}
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     x = df[quantitativeColumns[0]].values.ravel()
        #     # x = np.linspace(0,30,100)
        #     for k in [1, 2]:
        #         rv = chi2(k)
        #         pdf = rv.pdf(x)

        #         data['rv'].append(rv)
        #         data['pdf'].append(pdf)

        #         plt.plot(x, pdf, label="$k=%s$" % k)

        #     plt.legend()
        #     plt.title("PDF ($\chi^2_k$)")
        #     save_bytes_image(image_list)

        #     output_df = pd.DataFrame(data)

        #     res = {
        #         'output': output_df.head(10).to_json(orient='table'),
        #         'result': image_list,
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "plot-via-limit":
        #     df = self.current_df.select_dtypes(include='number')
        #     image_list = []
        #     p_steps = 100
        #     v_steps = 100

            
        #     # alt_data = df[df.columns[[0, 2]]]
        #     P = [item for sublist in df[df.columns[[0]]].values.tolist() for item in sublist]
        #     V = [item for sublist in df[df.columns[[1]]].values.tolist() for item in sublist]

        #     x = np.arange(-np.pi, np.pi, 2*np.pi/p_steps)
        #     y = []
        #     for i in range(len(x)): y.append([])

        #     for p, v in zip(P, V):
        #         i = int((p+np.pi)/(2*np.pi/p_steps))
        #         y[i].append(v)
                
        #     means = [ np.mean(np.array(vs)) for vs in y ]
        #     stds = [ np.std(np.array(vs)) for vs in y ]

        #     plt.plot(x, means)

        #     plt.plot([-2, -2], [-0.99, 0.99], 'k:', lw=1)
        #     plt.plot([2, 2], [-0.99, 0.99], 'k:', lw=1)

        #     plt.xlim([-np.pi, np.pi])
        #     plt.ylim([-1,1])
        #     plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        #             [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        #     plt.xlabel(r"Phase [rad]")
        #     plt.ylabel(r"V mean")
        #     save_bytes_image(image_list)
        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': image_list,
        #         'description' : description,
        #         'type': method
        #     }
        #     # self.intermediate_df.append(self.current_df)

        # elif method == "quantitative-bar-plot":
        #     from pandas.api.types import is_numeric_dtype

        #     df = self.current_df
        #     image_list = []
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     x = df[quantitativeColumns[0]].values.ravel()
        #     y = df[[quantitativeColumns[1]]].values.ravel()

        #     plt.figure()
        #     plt.title("Plot Bar")
        #     plt.bar(range(len(x)), y, align="center")
        #     plt.xticks(range(len(x)), rotation=90)
        #     plt.xlim([-1, len(x)])

        #     save_bytes_image(image_list)
        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': image_list,
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "outer-join":
        #     # needs two dataframes
        #     df1 = self.current_df
        #     res = {
        #         'output': df1.head(10).to_json(orient='table'),
        #         'result': df1.head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "word-to-vec":
        #     df = self.current_df

        #     res_df = calcWordVec(df)
        
        #     res = {
        #         'output': res_df.head(10).to_json(orient='table'),
        #         'result': res_df.head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)
        # elif method == "plot":
        #     df = self.current_df
        #     image_list = []
        #     import matplotlib.gridspec as gridspec

        #     samples = dict()
        #     alt_df = df.select_dtypes(include='number')
        #     h, w = alt_df.shape

        #     for a in np.arange(h):
        #             samples[a] = ((alt_df.iloc[[a]].values).ravel())[:4]
        #             if len(samples) >= 5:
        #                 break
        #             # print samples[a]
        #     # use only top 5 of the samples, otherwise that's too much
        #     fig = plt.figure(figsize=(10, 10))
        #     gs = gridspec.GridSpec(1, len(samples))
        #     gs.update(wspace=0.05, hspace=0.05)

        #     for i, sample in samples.iteritems():
        #         ax = plt.subplot(gs[i])
        #         plt.axis('off')
        #         ax.set_xticklabels([])
        #         ax.set_yticklabels([])
        #         ax.set_aspect('equal')
        #         plt.imshow(sample.reshape(2, 2), cmap='Greys_r')

        #     save_bytes_image(image_list)
        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': image_list,
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "demo-mat-show":
        #     df = self.current_df
        #     image_list = []

        #     from pandas.api.types import is_numeric_dtype
        #     from sklearn.metrics import confusion_matrix

        #     data = {'confusion': []}

        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     yy_test = df[quantitativeColumns[0]].values.ravel()
        #     yy_pred = df[quantitativeColumns[1]].values.ravel()
            
        #     confusion = confusion_matrix(yy_test, yy_pred)
            
        #     data['confusion'].append(confusion)

        #     # print(confusion)
        #     plt.matshow(confusion)
        #     plt.title('Confusion matrix')
        #     plt.gray()
        #     plt.ylabel('True label')
        #     plt.xlabel('Predicted label')
        #     save_bytes_image(image_list)

        #     plt.clf()
        #     # Ou si on voudrait que le noir nous montre les plus communs :
        #     invert_colors = np.ones(confusion.shape) * confusion.max()
        #     plt.matshow(invert_colors - confusion)
        #     plt.title('Confusion matrix')
        #     plt.gray()
        #     plt.ylabel('True label')
        #     plt.xlabel('Predicted label')
        #     save_bytes_image(image_list)

        #     res = {
        #         'output': image_list,
        #         'result': pd.DataFrame(data).head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "decision-tree-regressor":
        #     df = self.current_df

        #     from sklearn.tree import DecisionTreeRegressor
        #     from pandas.api.types import is_numeric_dtype
        #     tree_reg1 = DecisionTreeRegressor(random_state=42)
        #     quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        #     X = df[quantitativeColumns[:-1]]
        #     y = df[[quantitativeColumns[-1]]]

        #     tree_reg1.fit(X, y)
        #     y_pred1 = tree_reg1.predict(X)
        #     out_df = X.copy()
        #     out_df["Expected-"+quantitativeColumns[-1]] = y
        #     out_df["Predicted-"+quantitativeColumns[-1]] = y_pred1

        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': out_df.head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "initialize-kmeans-cluster":
        #     df = self.current_df

        #     res_df = initializeClustersForKmeans(df)
        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': res_df.head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)

        # elif method == "conditional-frequence-distribution":
        #     df = self.current_df

        #     res_df = calcConditionalFreqDist(df)
        #     res = {
        #         'output': df.head(10).to_json(orient='table'),
        #         'result': res_df.head(10).to_json(orient='table'),
        #         'description' : description,
        #         'type': method
        #     }
        #     self.intermediate_df.append(self.current_df)
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

# cluster id 162

def initializeClustersForKmeans(df):
    '''Use k-means++ to initialize a good set of centroids'''
    from pandas.api.types import is_numeric_dtype
    from sklearn.metrics import pairwise_distances

    k = 50
    quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
    centroids = np.zeros((k, len(quantitativeColumns)))
    data = df[quantitativeColumns].values

    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:]
    # Compute distances from the first centroid chosen to all the other data points
    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
        
    for i in xrange(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
        centroids[i] = data[idx,:]
        # Now compute distances from the centroids to all data points
        squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
        
    final = {}
    for i,c in enumerate(quantitativeColumns):
        final[c] = centroids[:,i]
    return pd.DataFrame(final)

def calcConditionalFreqDist(df):
    import nltk

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

# from keras.preprocessing.text import Tokenizer
# from gensim.models import word2vec

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
