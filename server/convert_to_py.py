import json
import os

with open('dictionary.json') as dictionary:
    data = json.load(dictionary)
    for block in data:
        #get code block content and file name for .py file
        block_name = block['user-data']['method']
        file_name = block_name+'.py'
        code_block_content = "".join(block['code']).decode('string_escape')

        full_file_path = os.path.join('./code_blocks', file_name)

        #write code block to the corresponding .py file
        f = open(full_file_path, "w+")
        f.write(code_block_content)

        f.close()
