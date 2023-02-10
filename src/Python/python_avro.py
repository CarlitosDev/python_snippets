import avro.schema

json_schema_string = '''
{
    "availability_id": "f90a2a7a-8769-4156-b2b9-72710e0463b9",
    "valid_from": "2019-08-04T23:00:00.000Z",
    "valid_to": "2020-08-03T23:00:00.000Z",
    "monday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "tuesday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "wednesday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "thursday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "friday": [
      {
        "from": "04:00:00",
        "to": "09:00:00"
      }
    ],
    "saturday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "sunday": [
      {
        "from": "03:30:00",
        "to": "13:00:00"
      }
    ],
    "timezone": "Europe/London"
}
'''

avro.schema.parse(json_schema_string)
avro.schema.Parse(json_schema_string)
avro.schema.SchemaFromJSONData(json_schema_string)
avro.schema.

####

import json

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

def create_schema():
    names = avro.schema.Names()
    load = lambda dict_value: avro.schema.SchemaFromJSONData(dict_value, names=names)

    transaction_schema_dict = {
        "namespace": "myavro",
        "type": "record",
        "name": "Transaction",
        "fields": [
            {"name": "name", "type": "string"},
        ]
    }
    account_schema_dict = {
        "namespace": "myavro",
        "type": "record",
        "name": "Account",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "transaction",  "type": ["null", {'type': 'array', 'items': 'Transaction'}], 'default': "null"},
        ]
    }

    load(transaction_schema_dict)
    return load(account_schema_dict)

def write_avro_file(file_path, schema, data):
    with open(file_path, 'wb') as f, DataFileWriter(f, DatumWriter(), schema) as writer:
        writer.append(data)

def print_avro_file(file_path):
    with open(file_path, 'rb') as f, DataFileReader(f, DatumReader()) as reader:
        for account in reader:
            print(account)


schema = create_schema()
file_path = 'account.avro'
data = {
    'name': 'my account',
    'transaction': [
        { 'name': 'my transaction 1' },
        { 'name': 'my transaction 2' },
    ]
}
write_avro_file(file_path, schema, data)
print_avro_file(file_path)



#####
json_blob = '''
    {
      "name": "XXXX",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "XXXX",
          "fields": [
            {
              "name": "from",
              "type": [
                    "null",
                    "string"
                  ],
                  "doc": "type of lesson",
                  "default": null,
                  "pii": false
            },
            {
                "name": "to",
                "type": [
                  "null",
                  "string"
                ],
                "doc": "TJ id for the booking",
                "default": null,
                "pii": false
            }
          ]
        }
      },
      "doc": "Array containing the start and end time of the for the day",
      "default": null,
      "pii": false
    }
    '''

weekdays = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']

all_blobs = []
for iDay in weekdays:
  all_blobs.append(json_blob.replace('XXXX', iDay))

import pyperclip
pyperclip.copy(','.join(all_blobs))