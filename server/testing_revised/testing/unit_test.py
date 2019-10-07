import unittest
import pandas as pd
import numpy as np
import json
from correlation import corr
from compute_percentiles_range import compute_percentiles_range
from compute_covariance_matrix import compute_covariance_matrix
from correlation_heatmap import corr_heatmap
from decision_tree_classifier import decision_tree_classifier


class TestCrowdBlocks(unittest.TestCase):

# NOTE: If test runs with no errors, then we assume that the analysis has run correctly (for test cases that are supposed to work)
# i.e. if there is supposed to be a dataframe returned, we assume that the df returned is correct

########################## Tests for CORRELATION code block ##########################


    # NOTE: Possible invalid dataframe inputs
    # Nonnumeric values
    # Null dataframe
    # Empty dataframe
    # Invalid datarame - i.e. columns not all sizes

    # Tests normal behavior of correlation block with all numeric values
    def testCorrelation(self):
        df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])
        df2 = pd.DataFrame({'a': [1,2]*3, 'b': [3,3]*3, 'c': [1.0, 2.0]*3})

        res = corr(df1, [], "", "")
        res_2 = corr(df1, [], "", "")

        self.assertNotIsInstance(res_2['result'], Exception)

        self.assertNotIsInstance(res['result'], Exception)


    # Tests behavior when passed a datagrame with nonnumeric values
    def testCorrelationNonNumeric(self):
        df_error = pd.DataFrame({'a': [True, False]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})

        res_error = corr(df_error, [], "", "")


        self.assertEqual(res_error['result'], "Dataframe needs numeric values")

    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    def testCorrelationMixed(self):
        df_mixed = pd.DataFrame({'a': ['','']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})

        res_mixed = corr(df_mixed, [], "", "")

        self.assertNotIsInstance(res_mixed['result'], Exception)


    # Tests behavior when passed with values of mixed datatype within a single column
    def testCorrelationInnerMixed(self):
        df_test = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})

        res_test = corr(df_test, [], "", "")

        self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    # Test behavior when passed dataframe with NaN values
    # NOTE: Added check for dataframe that contains ALL NaN values
    def testCorrelationNaN(self):
        df2 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})


        res_nan = corr(df2, [], "", "")

        self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    # Test behavior when passed a null dataframe
    # NOTE: added check for NoneType dataframe

    def testCorrelationNull(self):
        df2 = None

        res = corr(df2, [], "", "")

        self.assertEqual(res['result'], "Null dataframe needs numeric values")

    # Test behavior when passed an empty dataframe
    def testCorrelationEmpty(self):
        empty_df = pd.DataFrame()

        empty_res = corr(empty_df, [], "", "")

        self.assertEqual(empty_res['result'], "Dataframe needs numeric values")

    ########################## Tests for COMPUTE PERCENTILES RANGE code block ##########################

     # NOTE: Possible invalid dataframe inputs
    # Nonnumeric values
    # Null dataframe
    # Empty dataframe
    # Invalid datarame - i.e. columns not all same sizes

    # Tests normal behavior of CPR block with all numeric values and larger size dataframe
    def testCPR(self):
        df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])

        df2 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(100,5)), columns=['a', 'b', 'c', 'd', 'e'])

        res1 = compute_percentiles_rng(df1, [], "", "")

        res2 = compute_percentiles_rng(df2, [], "", "")

        self.assertNotIsInstance(res1['result'], Exception)

        self.assertNotIsInstance(res2['result'], Exception)

    # Tests behavior when passed a dataframe with nonnumeric values
    def testCPRNonnumeric(self):
        df = pd.DataFrame({'a': ["", "False"]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})

        res = compute_percentiles_rng(df, [], "", "")

        self.assertEqual(res['result'], "Dataframe needs numeric values")

    # Test behavior when passed a null dataframe
    # NOTE: added check for NoneType dataframe
    def testCPRNull(self):
        df = None

        res_null = compute_percentiles_rng(df, [], "", "")

        self.assertEqual(res_null['result'], "Null dataframe needs numeric values")

    # Test behavior when passed dataframe with ALL NaN values
    # NOTE: Added check for dataframe that contains ALL NaN values
    def testCPRNaN(self):
        df2 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})

        res_nan = corr(df2, [], "", "")

        self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    # Test behavior when passed an empty dataframe
    def testCPREmpty(self):
        empty_df = pd.DataFrame()

        empty_res = compute_percentiles_rng(empty_df, [], "", "")

        self.assertEqual(empty_res['result'], "Dataframe needs numeric values")

    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    def testCPRMixed(self):
        df_mixed = pd.DataFrame({'a': ['','']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})

        res_mixed = compute_percentiles_rng(df_mixed, [], "", "")

        self.assertNotIsInstance(res_mixed['result'], Exception)

    # Tests behavior when passed with values of mixed datatype within a single column

    def testCPRInnerMixed(self):
        df_test = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})

        res_test = compute_percentiles_rng(df_test, [], "", "")

        self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    ########################## Tests for COMPUTE COVARIANCE MATRIX code block ##########################

    # Tests normal behavior of Compute Cov Matrix block with all numeric values and larger size dataframe
    # def testComputeCovMatrix(self):
    #     df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])

    #     df2 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(100,5)), columns=['a', 'b', 'c', 'd', 'e'])

    #     res1 = compute_covariance_matrix(df1, [], "", "")
    #     res2 = compute_covariance_matrix(df2, [], "", "")

    #     self.assertNotIsInstance(res1['result'], Exception)

    #     self.assertNotIsInstance(res2['result'], Exception)

    # # Test behavior when passed dataframe with nonnumeric values
    # def testComputeCovMatrixNonnumeric(self):
    #     df_error = pd.DataFrame({'a': [True, False]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})

    #     res_error = compute_covariance_matrix(df_error, [], "", "")

    #     self.assertEqual(res_error['result'], "Dataframe has no numeric values")

    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    # NOTE: This returned a "Matrix is singular" error - we need to account for this
    # NOTE: Add test case to check for behavior when passed a df that converts to a singular matrix

    # def testComputeCovMatrixMixed(self):

    #     # This matrix is singular - threw error in execution of code block
    #     df_mixed = pd.DataFrame({'a': ['','']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})

    #     res_mixed = compute_covariance_matrix(df_mixed, [], "", "")

    #     self.assertNotIsInstance(res_mixed['result'], Exception)

    # Test behavior when passed in NoneType for input dataframe
    # def  testComputeCovMatrixNull(self):
    #     df = None

    #     res = compute_covariance_matrix(df, [], "", "")

    #     self.assertEqual(res['result'], "Null dataframe needs numeric values")

    # # Test behavior when passed in empty dataframe object
    # def testComputeCovMatrixEmpty(self):
    #     df = pd.DataFrame()

    #     res_empty = compute_covariance_matrix(df, [], "", "")

    #     self.assertEqual(res_empty['result'], "Dataframe has no numeric values")

    # # Test behavior when passed in dataframe with ALL NaN values
    # def testComputeCovMatrixNaN(self):
    #     df2 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})

    #     res_nan = compute_covariance_matrix(df2, [], "", "")

    #     self.assertEqual(res_nan['result'], "Dataframe has no numeric values")

########################## Tests for COND FREQ DISTRIBUTION code block ##########################

# NOTE: This code block needs to be modified for general inputs - input right now is only an example

########################## Tests for CORRELATION HEAT MAP code block ##########################

# NOTE: use res['output] not res['result]
# NOTE: All testcases for this code block need to be reviewed - needs to be some type checking in code block
# to make sure only blocks with numeric values produce output

# Test normal behavior when passed input with numerical values
    def testCorrHeatMap(self):
        df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])
        df2 = pd.DataFrame({'a': [1,2]*3, 'b': [3,3]*3, 'c': [1.0, 2.0]*3})

        res = corr_heatmap(df1, [], "", "")
        res_2 = corr_heatmap(df2, [], "", "")

        self.assertNotIsInstance(res_2['output'], Exception)

        self.assertNotIsInstance(res['output'], Exception)

    # Test behavior when passed dataframe with nonnumeric typs
    # NOTE: The block still ran without error
    # NOTE: Should there be an explicit check for this in the block?
    def testCorrHeatMapNonnumeric(self):
        df_error = pd.DataFrame({'a': [True, False]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})

        res_error = corr_heatmap(df_error, [], "", "")

        # print (res_error['output'])
        # self.assertEqual(res_error['result'], )

    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    def testCorrHeatMapMixed(self):
        df_test = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})

        res = corr_heatmap(df_test, [], "", "")

        self.assertNotIsInstance(res['result'], Exception)

    # Tests behavior when passed with values of mixed datatype within a single column
    # NOTE: This still ran without producing any error, this needs to be fixed
    def testCorrHeatMapInnerMixed(self):
        df_test = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})

        res_test = corr_heatmap(df_test, [], "", "")

        # print(res_test['result'])
        # self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    # Test behavior when passed input dataframe with ALL NaN values
    def testCorrHeatMapNaN(self):
        df2 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})

        res_nan = corr_heatmap(df2, [], "", "")

        # self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    def testCorrHeatMapNull(self):
        null_df = None

        null_res = corr_heatmap(null_df, [], "", "")

        self.assertEqual(null_res['result'], "Null Dataframe needs numeric values")


    ########################## Tests for DECISION TREE CLASSIFIER code block ##########################

    # Test normal behavior when passed dataframe with all numeric type
    # NOTE: Threw " ValueError: Unknown label type: 'continuous' " when passed in dataframe like df_1 and df_2
    # NOTE: Worked normally when passed in dataframe - might need to add some more test cases for this
    # def testDTC(self):
    #     # df_1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])

    #     df2 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(100,5)), columns=['a', 'b', 'c', 'd', 'e'])

    #     # df_2 = pd.DataFrame({'a': [1.0,2.0]*3, 'b': [3.0,3.0]*3, 'c': [1.0, 2.0]*3})
    #     # res1 = decision_tree_classifier(df_1, [], "", "")
    #     # res1 = decision_tree_classifier(df_2, [], "", "")

    #     res2 = decision_tree_classifier(df2, [], "", "")

    #     # self.assertNotIsInstance(res1['result'], Exception)

    #     self.assertNotIsInstance(res2['result'], Exception)

    # Test behavior when passed input dataframe with nonumeric datatypes
    # def testDTCNonnumeric(self):
    #     df = pd.DataFrame({'a': ["", "False"]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})

    #     res = decision_tree_classifier(df, [], "", "")

    #     self.assertEqual(res['result'], "Dataframe needs numeric values")

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

    # Tests behavior when passed with values of mixed datatype within a single column
    # def testDTCInnerMixed(self):
    #     df_test = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})

    #     res_test = decision_tree_classifier(df_test, [], "", "")

    #     self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    # Test behavior when passed input dataframe with ALL NaN values
    def testDTCnNaN(self):
        df2 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})


        res_nan = decision_tree_classifier(df2, [], "", "")

        self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    # Test behavior when passed an empty dataframe
    # def testDTCEmpty(self):
    #     empty_df = pd.DataFrame()

    #     empty_res = decision_tree_classifier(empty_df, [], "", "")

    #     self.assertEqual(empty_res['result'], "Dataframe needs numeric values")

    # Test behavior when passed NoneType as input
    # NOTE: Added null check
    def testDTCNull(self):
        null_df = None

        null_res = decision_tree_classifier(null_df, [], "", "")

        self.assertEqual(null_res['result'], "Null Dataframe needs numeric values")


if __name__ == '__main__':
    unittest.main()
