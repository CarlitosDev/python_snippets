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





# Explode columns in a DF that are json
df_exploded_json = pd.json_normalize(df_current_event['@kafka.body'].apply(lambda s: json.loads(s)))




##### Example of using path
json_blob = '''{
  "needGrading": true,
  "totalAttendanceTime": 2549,
  "studentNeedRecord": false,
  "grantUsageId": 0,
  "roomId": "ff7d4586-53b2-4dcf-9fbb-774d7bf35876",
  "studentId": "5f3e953f-6186-4b62-a0f0-da96ac363501",
  "classId": 23244878,
  "topicId": 677,
  "teacherMemberId": 1319988,
  "teacherAttendanceList": [
    {
      "exitTime": "2020-09-11T11:45:19.72Z",
      "userId": "1319988",
      "enterTime": "2020-09-11T10:59:31.2Z"
    }
  ],
  "serviceSubTypeCode": "Global",
  "topic": "Greetings and introductions",
  "startTime": "2020-09-11T11:00:00Z",
  "studentAttendanceList": [
    {
      "exitTime": "2020-09-11T11:44:59.713Z",
      "userId": "5f3e953f-6186-4b62-a0f0-da96ac363501",
      "enterTime": "2020-09-11T11:02:30.687Z"
    }
  ],
  "serviceTypeCode": "PL",
  "statusCode": "Attended",
  "schoolCode": "b2c"
}'''
json_dict = json.loads(json_blob)

vars_to_explode = [k for k,v in json_dict.items() if isinstance(v, list)]

col_names = [*json_dict.keys()]
meta_col_names = [*set(col_names) - set(vars_to_explode)]

df_exp_A = pd.json_normalize(data=json_dict, record_path=vars_to_explode[0], 
                            meta=meta_col_names)

df_exp_B = pd.json_normalize(data=json_dict, record_path=vars_to_explode[1], 
                            meta=['studentId'])

df = pd.merge(df_exp_A, df_exp_B, on='studentId')
df.iloc[0]