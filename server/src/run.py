from flask import Flask, request, Response, send_file, jsonify

import os
import pandas as pd
import numpy as np

import ujson
import json
import time
import logging

from analysis import Analysis
from recommender import Recommender


app = Flask(__name__)

# disable logging for data transfer
log = logging.getLogger('werkzeug')
log.disabled = True

analysis = Analysis()
recommender = Recommender()
data_read = pd.DataFrame()

@app.route('/datasets')
def scan_dataset():
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/../../data/'
    files = [f.split('.')[0] for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    return ujson.dumps(files)



@app.route('/dataload')
def data_load():
    if request.args['dataset']:
        data_read = load_dataset(request.args['dataset'])
    else:
        data_read = pd.read_csv(request.args['data'], delimiter=',')

    return 'Data Loaded!'

# @app.route('/calculate')
# def calculate_intermediate(): 
#     if request.args['compute'] == 'suggest':
#         state = None if not request.args['state'] else request.args['state'] 
#         # intermediate_idx = int(request.args['index'])
#         # selected_df = analysis.intermediate_df[intermediate_idx]
#         # analysis.current_df = selected_df; 
#         expert_suggestions = recommender.get_manual_suggestions(state)
#         crowd_suggestions = recommender.get_crowd_suggestions(state)
#         all_analysis = recommender.get_analysis_list()
#         res = {
#             'all_analysis': all_analysis,
#             'manual_result' : expert_suggestions,
#             'crowd_result' : crowd_suggestions,
#             'type' : 'suggest'
#         }
#     else: 
#         # intermediate_idx = int(request.args['index'][-1])
#         # selected_df = analysis.intermediate_df[intermediate_idx]

#         print (request.args['dataframe'].__class__)
#         print (request.args['dataframe'])
#         analysis.intermediate_selected = True
#         analysis.current_df = pd.read_json(request.args['dataframe'])
#         print("current_df")
#         print(analysis.current_df)
#         # print ("All dfs") 
#         # print (analysis.intermediate_df)
#         # print ("LENGTH")
#         # print (len(analysis.intermediate_df))
#         result = analysis.execute_analysis(request.args['compute'], request.args['dataset'], request.args['state'])
#         return ujson.dumps(result)


@app.route('/compute')
def compute():
    if request.args['compute'] == 'suggest':
        state = None if not request.args['state'] else request.args['state']
        expert_suggestions = recommender.get_manual_suggestions(state)
        crowd_suggestions = recommender.get_crowd_suggestions(state)
        

        # run suggested code blocks to see if an error is thrown, if so, do not include in list of expert/crowd
        # shown to user
        # to_remove = []
        # for i,cr_analysis in enumerate(crowd_suggestions):
        #     temp_res = analysis.execute_analysis(str(cr_analysis), request.args['dataset'], state)
        #     if temp_res['type'] == 'error':
        #         to_remove.append(i)
        #     else: 
        #         analysis.intermediate_df.remove(analysis.intermediate_df[-1])
        
        # print ("Remove: ", to_remove)
        # crowd_suggestions = [crowd_suggestions[i] for i in range(len(crowd_suggestions)) if i not in to_remove]

        # for ex_analysis in expert_suggestions:
        #     # print(typeof(ex_analysis))
        #     # print(ex_analysis)
        #     temp_res = analysis.execute_analysis(str(ex_analysis), request.args['dataset'], state)
        #     if temp_res['type'] == 'error':
        #         # print("EXPERT_ERROR") 
        #         expert_suggestions.remove(ex_analysis)
        #     else: 
        #         analysis.intermediate_df.remove(analysis.intermediate_df[-1])

        # for some reason this had to be run again - should clean this up later - noticed that there were still 
        # some recs producing errors even after first check
        # for cr_analysis in crowd_suggestions:
        #     # print(typeof(cr_analysis))
        #     # print(cr_analysis)
        #     temp_res = analysis.execute_analysis(str(cr_analysis), request.args['dataset'], state)
        #     if temp_res['type'] == 'error':
        #         # print("CROWD_ERROR") 
        #         crowd_suggestions.remove(cr_analysis)
        #     else: 
        #         analysis.intermediate_df.remove(analysis.intermediate_df[-1])
        
        
        all_analysis = recommender.get_analysis_list()
        res = {
            'all_analysis': all_analysis,
            'manual_result' : expert_suggestions,
            'crowd_result' : crowd_suggestions,
            'type' : 'suggest'
        }
        return ujson.dumps(res)

    else:
        # print (request)
        # print (request.args['dataframe'].__class__)
        # print (request.args['dataframe'])
        # # analysis.current_df = pd.read_json(request.args['dataframe'])

        # print("Number of steps:") 
        # print (request.args['steps'].__class__)
        # print (request.args['steps'])
        analysis.intermediate_selected = True

        # if len(analysis.intermediate_df) != 0:
        #     analysis.current_df = analysis.intermediate_df[-1]

        # print("current_df")
        # print(analysis.current_df)
        result = analysis.execute_analysis(request.args['compute'], request.args['dataset'], request.args['state'])
        return ujson.dumps(result)


@app.route('/intermediatedata')
def delete():
    res = {
        'type': request.args['type']
    }
    if request.args['type'] == 'delete':
        analysis.delete_intermediate_df(request.args['index'])
    elif request.args['type'] == 'export':
        res['data'] = analysis.export_intermediate_df(request.args['index']).to_json(orient='records')

    return ujson.dumps(res)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


def load_dataset(filename):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/../../data/', filename + ".csv")
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
