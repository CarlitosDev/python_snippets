from pandas.io.json import json_normalize
import pandas as pd
import json

json_blob = '''{
    "range": [
        {
            "begin": "2019-04-03T10:20:37Z",
            "end": "2019-04-03T10:20:37Z"
        }, 
        {
            "begin": "2019-03-03T10:20:37Z",
            "end": "2019-04-01T10:20:37Z"
        }
    ],
    "teachers": [
        "07687728-b62c-4e7f-95fe-e64f9bf68927",
        "09687728-b62c-4e7f-95fe-a876a68f83a7"
    ],
    "fake_id" : "43124"
}'''
import json
json_dict  = json.loads(json_blob)




works_data = json_normalize(data = json_dict, 
record_path ='range',
meta =['fake_id'])
works_data.head()

works_data = json_normalize(data = json_dict, 
record_path ='teachers',
meta =['fake_id'])
works_data.head()