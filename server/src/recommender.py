import pandas as pd
import numpy as np
import os
import json
import random

crowd_analysis_order = 'new_crowd_analysis_order.json'

class Recommender:
    def __init__(self, state=None):
        self.state = state
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + crowd_analysis_order, 'r') as f:
            self.json_data = json.load(f)
            self.crowdIds = self.getCrowdIds()
            self.expertIds = self.getExpertIds()
        

    #modify so it interacts with dictionary and returns suggestions
    #ranks list of entries its been given from input dictionary or list of templates/codeblocks
    #change this to use dictionary
    def get_manual_suggestions(self, state=None):
        suggestion_dict = dict()

        # the name of the most recently executed block
        current_analysis = state
        current_analysis = json.loads(current_analysis)

        # do we have a name?
        if current_analysis:
          # get the id
          current_id = self.getIdForBlockName(current_analysis)
          # this recommendation was from the crowd, need something from the experts
          if current_id and current_id not in self.expertIds:
            # get the relevant expert ids
            candidates = self.retrieveExpertBlockIdsByBlockTags(current_id)
            # do we have any expert blocks with overlapping tags?
            if candidates and len(candidates) > 0:
              # randomly pick a block from the list for now
              new_id = candidates[random.randint(0,len(candidates)-1)]
              # get the corresponding block name
              current_analysis = self.codeblock_id_to_name(new_id)

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'manual_analysis_order.json', 'r') as f:
            json_data = json.load(f)
        
        for i in range(0, len(json_data)):
            tutorial_order = json_data[i]["analysis-order"]

            for index, step in enumerate(tutorial_order):
                if current_analysis:
                    if step == current_analysis and index <  len(tutorial_order) - 1:
                        if tutorial_order[index + 1] not in suggestion_dict:
                            suggestion_dict[tutorial_order[index + 1]] = [1, self.get_description(tutorial_order[index + 1])]
                        else:
                            curr_val = suggestion_dict[tutorial_order[index + 1]][0]
                            suggestion_dict[tutorial_order[index + 1]][0] = curr_val + 1
                
                else:
                    if tutorial_order[0] not in suggestion_dict:
                        suggestion_dict[tutorial_order[0]] = [1, self.get_description(tutorial_order[0])]
                    else:
                        curr_val = suggestion_dict[tutorial_order[0]][0]
                        suggestion_dict[tutorial_order[0]][0] = curr_val + 1

        s = sum([pair[0] for pair in suggestion_dict.values()])
        for k, v in suggestion_dict.items():
            suggestion_dict[k][0] = str(round(v[0] * 100 / s, 1)) + '%'

        sorted_dict = sorted(suggestion_dict.items(), key=lambda kv: v[0], reverse=True)
        suggestions = [{'name': item[0], 'probability': item[1][0], 'description': item[1][1]}  for item in sorted_dict]
        print(suggestions)

        return suggestions

    def get_crowd_suggestions(self, state=None):
        suggestion_dict = dict()
        suggestions = []
        current_analysis = state
        current_analysis = json.loads(current_analysis)
        current_id = "-1"

        # do we have a name?
        if current_analysis:
          #print "original current_analysis",current_analysis
          # get the id
          current_id = self.getIdForBlockName(current_analysis)
          # this recommendation was from the experts, need something from the crowd
          if current_id and current_id not in self.crowdIds:
            # get the relevant crowd ids
            candidates = self.retrieveCrowdBlockIdsByBlockTags(current_id)
            #print "candidates:",candidates
            # do we have any crowd blocks with overlapping tags?
            if candidates and len(candidates) > 0:
              # randomly pick a block from the list for now
              new_id = candidates[random.randint(0,len(candidates)-1)]
              # get the corresponding block name
              current_analysis = self.codeblock_id_to_name(new_id)
              #print "new current_analysis",current_analysis

        # automated analysis needs one more step from name code block name to id
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f_dict:
            json_dict = json.load(f_dict)

        for i in range(0, len(json_dict)):
            if current_analysis in json_dict[i]["user-data"]["method"]:
                current_id = json_dict[i]["id"]

        print(current_id)

        for notebook_id, tutorial_order in self.json_data.items():

            for index, step in enumerate(tutorial_order):
                if current_analysis:
                    if str(step) == current_id and index <  len(tutorial_order) - 1:
                        next_analysis = self.codeblock_id_to_name(str(tutorial_order[index + 1]))
                        if next_analysis:
                            if next_analysis not in suggestion_dict:
                                suggestion_dict[next_analysis] = [1, self.get_description(next_analysis)]
                            else:
                                curr_val = suggestion_dict[next_analysis][0]
                                suggestion_dict[next_analysis][0] = curr_val + 1
                
                else:
                    next_analysis = self.codeblock_id_to_name(str(tutorial_order[0]))
                    if next_analysis:
                        if next_analysis not in suggestion_dict:
                            suggestion_dict[next_analysis] = [1, self.get_description(next_analysis)]
                        else:
                            curr_val = suggestion_dict[next_analysis][0]
                            suggestion_dict[next_analysis][0] = curr_val + 1

        s = sum([pair[0] for pair in suggestion_dict.values()])
        for k, v in suggestion_dict.items():
            suggestion_dict[k][0] = str(round(v[0] * 100 / s, 1)) + '%'

        sorted_dict = sorted(suggestion_dict.items(), key=lambda kv: v[0], reverse=True)
        suggestions = [{'name': item[0], 'probability': item[1][0], 'description': item[1][1]}  for item in sorted_dict]
        print(suggestions)
        return suggestions

    def get_analysis_list(self):
        analyses = []

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            analyses.append({'name': json_data[i]["user-data"]["method"], 'desecription': json_data[i]["description"]})

        return analyses

    def get_description(self, name):
        description = ""
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            if json_data[i]['user-data']['method'] == name:
                description = json_data[i]['description']
                break

        return description

    # for the given block id, get the ids of blocks with the same tags
    def retrieveCrowdBlockIdsByBlockTags(self,blockId,blocks=None):
      blockId = str(blockId)
      tags = self.getTagsForBlockId(blockId,blocks)
      return self.retrieveBlockIdsForTags(tags,self.crowdIds.values())

    # for the given block id, get the ids of blocks with the same tags
    def retrieveExpertBlockIdsByBlockTags(self,blockId,blocks=None):
      blockId = str(blockId)
      tags = self.getTagsForBlockId(blockId,blocks)
      return self.retrieveBlockIdsForTags(tags,self.expertIds.values())

    # for the given block id, retrieve the corresponding tags
    def getTagsForBlockId(self,blockId,blocks=None):
      blockId = str(blockId)
      if not blocks:
        blocks = self.openDictionary().values()
      for block in blocks:
        if blockId == block["id"]:
          return block["user-data"]["tags"]
      return None

    # get the names of the blocks that match any of the tags
    def retrieveBlockNamesForTags(self, tags,blocks=None):
      idxs = self.retrieveBlockIdsForTags(tags,blocks)
      #print "indexes:",idxs
      #print "methods:",[self.codeblock_id_to_name(idx) for idx in idxs]
      return [self.codeblock_id_to_name(idx) for idx in idxs]

    # get the identifiers of the blocks that match any of the tags
    def retrieveBlockIdsForTags(self,tags,blocks=None):
      relevantIds = {}
      if not blocks:
        blocks = self.openDictionary().values()
      for tag in tags:
        for block in blocks:
          if tag in block["user-data"]["tags"]:
            relevantIds[block["id"]] = block
      #print "indexes:",relevantIds.keys()
      return relevantIds.keys()

    # get all ids associated with expert recommendations
    def getExpertIds(self):
      allBlocks = self.openDictionary()
      #print "total blocks:",len(allBlocks)
      expertIds = {}
      with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'manual_analysis_order.json', 'r') as f:
        expertOrderings = [o["analysis-order"] for o in json.load(f)]
        for ordering in expertOrderings:
          for idx in ordering:
            if idx in allBlocks:
              expertIds[idx] = allBlocks[idx]
      return expertIds

    # get all ids associated with crowd recommendations
    def getCrowdIds(self):
      allBlocks = self.openDictionary()
      #print "total blocks:",len(allBlocks)
      crowdIds = {}
      with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + crowd_analysis_order, 'r') as f:
        crowdOrderings = (json.load(f)).values()
        for ordering in crowdOrderings:
          for idx in ordering:
            idx = str(idx)
            if idx in allBlocks:
              crowdIds[idx] = allBlocks[idx]
      return crowdIds

    # open the given or default dictionary file
    def openDictionary(self,filename=None):
      if not filename:
        filename = os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json'
      d = None
      with open(filename) as f:
        d = json.load(f)
      dmap = {}
      for b in d:
        dmap[b["id"]] = b
      return dmap

    # for the given block name return the id of the first matching block
    def getIdForBlockName(self,blockName,blocks=None):
      if not blocks:
        blocks = (self.openDictionary()).values()
      for block in blocks:
        if blockName == block["user-data"]["method"]:
          return block["id"]
      return None

    def codeblock_id_to_name(self, block_id):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f_dict:
            json_dict = json.load(f_dict)
        
        name = ""

        for i in range(0, len(json_dict)):
            if block_id == json_dict[i]["id"]:
                name = json_dict[i]["user-data"]["method"]

        return name

if __name__ == "__main__":
  rec = Recommender()
  #print rec.get_manual_suggestions("\"unique-column-values\"")
  #print rec.get_crowd_suggestions("\"fit-decision-tree\"")
  print(rec.get_crowd_suggestions("\"group-statistics\""))
