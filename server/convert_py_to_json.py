import json
import os

def loadCodeMap(cmfile):
    dic = []
    with open(cmfile) as openCFile:
        cmap = json.load(openCFile)

        for oneMap in cmap:
            blockPath = oneMap["path"]
            id = oneMap["id"]

            currJson = convert_py_to_json(os.path.join('code_blocks', blockPath))
            currJson["id"] = id
            dic.append(currJson)

    print(dic)
    return dic

def modifyDictionary(dFile, dJson):
    modifiedDictionary = []
    with open(dFile) as openDFile:
        dmap = json.load(openDFile)

        for oldJson in dmap:
            for newJson in dJson:
                if (newJson["id"] == oldJson["id"]):
                    currentJson = {}
                    currentJson["id"] = oldJson["id"]
                    currentJson["code"] = newJson["code"]
                    currentJson["metadata"] = oldJson["metadata"]
                    currentJson["description"] = oldJson["description"]
                    currentJson["user-data"] = oldJson["user-data"]
                    modifiedDictionary.append(currentJson)

    with open('dictionary_tmp.json', 'w') as outfile:
        json.dump(modifiedDictionary, outfile)


def convert_py_to_json(pyfile):
    code = []
    with open(pyfile) as pfile:
        for line in pfile:
            code.append(line)

    code_json = {'code': code}
    print(code_json)

    return code_json

dJson = loadCodeMap("dictionary_code_map.json")
modifyDictionary('dictionary.json', dJson)
#convert_py_to_json(os.path.join('code_blocks', 'group-statistics.py'))