import unittest
import pandas as pd
import numpy as np
import seaborn as sns
import json
import re
import ast
from codeblock_class import CodeBlock

class TestCrowdBlocks(unittest.TestCase):
    df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])
    df2 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [6, 'z', 8]})
    df3 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [6, np.nan, 8]})
    df4 = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")
    df5 = pd.DataFrame({'a': ["X", "Y", "Z"], 'b': ["T", "U", "V"]})
    df6 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [np.nan, np.nan, 8]})
    df7 = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12], [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
    columns=['Apple', 'Orange', 'Banana', 'Pear'], index=['Basket1', 'Basket2', 'Basket3', 'Basket4', 'Basket5', 'Basket6'])
    df8 = pd.DataFrame({'a': [True, False]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})
    df9 = pd.DataFrame({'a': ['','']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
    df10 = None

############################ Helper Functions ##################################

    # strip and reformat res['result'] into a dataframe
    def converter(self, obj):
        # gives back a string representation of the data
        data = obj.partition("data\":")[2]

        # get row names
        indices = [m.start() for m in re.finditer('index', data)]
        lst = map(lambda x : x + 8, indices)
        row_names = map(lambda x : data[x:data.find('\"', x)], lst)

        # get data values
        raw_cols = re.findall('\{(.*?)\}', data)
        cols = map(lambda x : x[x.index(",") + 1:], raw_cols)
        cols = map(lambda x : "{" + x + "}", cols)
        cols_data = map(lambda x : ast.literal_eval(x), cols)

        # create the dataframe
        return pd.DataFrame(cols_data, row_names)

    # return True if a df has only numeric values and False otherwise
    def numeric_check(self, df):
        lst = df.apply(lambda s : pd.to_numeric(s, errors='coerce').notnull().all()).to_string()
        return False if "False" in lst else True


############################### Tests ##########################################

    # add this to the real anova file: from scipy import stats
    def test_anova(self):
        res = anova_variance(self.df1, [], "", "")
        self.assertEqual(res['result'], "Dataframe contained incorrect values")

        res = anova_variance(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        # returns an error, but our error checking might be too strict
        # we don't want to throw out a whole analysis because of one nan value
        res = anova_variance(self.df3, [], "", "")
        #print(res['result'])

        res = anova_variance(self.df4, [], "", "")
        self.assertTrue(res['result'])

    def test_bootstrap(self):
        res = bootstrap(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

        res = bootstrap(self.df2, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

        res = bootstrap(self.df5, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_category_boxplot(self):
        res = cat_boxplot(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        res = cat_boxplot(self.df4, [], "", "")
        self.assertTrue(res['result']) # this is an image, so for now just check if it runs and returns and image_list

    def test_category_count(self):
        res = cat_count(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        res = cat_count(self.df4, [], "", "")
        self.assertTrue(res['result'])

    def test_compute_covariance_matrix(self):
        res = compute_covariance_matrix(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        res = compute_covariance_matrix(self.df1, [], "", "")
        self.assertEqual(res['result'], "Matrix is singular")

        # should this really give "matrix is singular" for df7?
        res = compute_covariance_matrix(self.df7, [], "", "")
        # print(res['result'])

    ###################### CORRELATION Tests ###########################
    def test_corr(self):
        res = corr(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 9)
        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.shape[1], 3)
        self.assertTrue(self.numeric_check(df))

    def test_corr_nonnumeric(self):
        res = corr(self.df8, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_mixed(self):
        res = corr(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_corr_inner_mixed(self):
        res = corr(self.df2, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_nan(self):
        res = corr(self.df6, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_null(self):
        res = corr(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null dataframe needs numeric values")

    ################## COMPUTE PERCENTILES RANGE Tests #######################
    def test_cpr(self):
        res = compute_percentiles_rng(self.df1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_cpr_nonnumeric(self):
        res = compute_percentiles_rng(self.df8, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_cpr_mixed(self):
        res = compute_percentiles_rng(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_cpr_inner_mixed(self):
        res = compute_percentiles_rng(self.df2, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_cpr_nan(self):
        res = compute_percentiles_rng(self.df6, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_cpr_null(self):
        res = corr(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null dataframe needs numeric values")

    ################### CORRELATION HEAT MAP Tests ############################
    def test_corr_heatmap(self):
        res = corr_heatmap(self.df1, [], "", "")
        self.assertNotIsInstance(res['output'], Exception)

    # NOTE: The block still ran without error
    # NOTE: Should there be an explicit check for this in the block?
    def test_corr_heatmap_onnumeric(self):
        res = corr_heatmap(self.df8, [], "", "")

        # print (res_error['output'])

    def test_corr_heatmap_mixed(self):
        res = corr_heatmap(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)


    # NOTE: This still ran without producing any error, this needs to be fixed
    def test_corr_heatmap_inner_mixed(self):
        res = corr_heatmap(self.df2, [], "", "")

        # print(res_test['result'])
        # self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    def test_corr_heatmap_nan(self):
        res = corr_heatmap(self.df6, [], "", "")
        # self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    def testCorrHeatMapNull(self):
        res = corr_heatmap(self.df10, [], "", "")
        self.assertEqual(null['result'], "Null Dataframe needs numeric values")

################### DECISION TREE CLASSIFIER Tests #############################
    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    # NOTE: Modified block to check for <= 1 quantitative columns
    # NOTE: Ran with 1 numeric column - threw shape error
    # NOTE: Ran with 2 numeric column - did not throw error - maybe we need to have atleast two numeric columns
    def testDTCMixedInvalid(self):
        df_mixed = pd.DataFrame({'a': ['10','1']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
        res_mixed = decision_tree_classifier(df_mixed, [], "", "")
        self.assertEqual(res_mixed['result'], "Dataframe needs numeric values")

    def testDTCMixedValid(self):
        df_mixed = pd.DataFrame({'a': [10,1]*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
        res_mixed = decision_tree_classifier(df_mixed, [], "", "")
        self.assertNotIsInstance(res_mixed['result'], Exception)

    def test_dtc_null(self):
        res = decision_tree_classifier(self.df10, [], "", "")
        self.assertEqual(null_res['result'], "Null Dataframe needs numeric values")



    def test_nan_cols(self):
        res = drop_cols(self.df3, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[1], 2)

        res = drop_cols(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[1], 3)

    def test_nan_rows(self):
        res = drop_rows(self.df1, [], "", "")
        self.assertEqual(res['description'], "Dataframe has no rows with NaN entries")

        res = drop_rows(self.df3, [], "", "")
        self.assertTrue(res['result'])

    def test_mean(self):
        res = mean(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 3)

        res = mean(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_first_ten(self):
        res = firstTen(self.df4, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[0], 10)

        res = firstTen(self.df2, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 9)

    def test_top_five(self):
        res = top5cat(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        # this returns an empty df, but i think it shouldn't
        res = top5cat(self.df7, [], "", "")
        df = self.converter(res['result'])

    def test_des(self):
        res = des(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

        res = des(self.df3, [], "", "")
        self.assertTrue(res['result'])

    def test_variance(self):
        res = variance(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[0], 4)

        res = variance(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_unique_cols(self):
        res = unique_column_values(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        res = unique_column_values(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)
        self.assertEqual(df.size, 30)

        res = unique_column_values(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertLess(df.size, self.df7.size)

    # add: import pandas as pd to actual test_linear_regression.py file
    def test_lin_reg(self):
        # what df would work? df1 and df7 don't work
        res = test_linear_regression(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)
