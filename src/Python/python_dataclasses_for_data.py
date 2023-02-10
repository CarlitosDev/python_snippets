'''
  From 
  https://dev.to/furkan_kalkan1/a-fully-automated-metadata-objects-with-python-37s-brand-new-dataclass-library-492i
  and
  https://dev.to/btaskaya/dataclasses-in-python-4hli


Dataclasses are a standard library module (that added in Python 3.7)
for code generation with given type annotataed specs. 

It simplifies the process of writing a class as a mutable
data holder with automatically generated ordering methods.

'''


import random, json
from dataclasses import dataclass, asdict, field
import datetime as dt
from typing import List
import os

@dataclass
class dataContainer:
    data_type: str
    file_location: str
    creation_time: dt.datetime = dt.datetime.utcnow().\
        replace(tzinfo=dt.timezone.utc).isoformat()
    description: str = 'Not provided'
    producer: str = 'Not provided'
    consumer: str = 'Not provided'
    signature: str = 'dataContainer'
    version: str = '0.0'
    list_of_files: List[str] = field(default_factory=list)
    def to_json(self):
        return json.dumps(asdict(self), indent=4, default=str)
    def save_json(self, json_file=''):
        json_blob = self.to_json()
        self.writeTextFile(json_blob, json_file)
    @staticmethod
    def readJSONFile(thisFile):
        with open(thisFile, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    @staticmethod
    def writeTextFile(thisStr, thisFile):
        with open(thisFile, 'w') as f:
            f.write(thisStr)
    @classmethod
    def from_json(cls, json_file):
        json_info = dataContainer.readJSONFile(json_file)
        data_type = json_info['data_type']
        file_location = json_info['file_location']
        self = cls(data_type, file_location)
        self.creation_time = json_info['creation_time']
        self.producer = json_info['producer']
        self.consumer = json_info['consumer']
        self.signature = json_info['signature']
        self.version = json_info['version']
        self.list_of_files = json_info['list_of_files']
        return self



bucket_name    = 'tod-ingest-segment-data'
key_prefix     = 'segment-logs'

data_type = 'aws_s3_bucket'
file_location = f's3://{bucket_name}/{key_prefix}/'
description = 'Tester for the Segment data'

# Instanciate the class
dc_Segment = dataContainer(data_type, file_location, description = description)
dc_Segment.producer = 'Segment APP'
json_file = '/Users/carlos.aguilar/Documents/stupid_test/container_1.json'
dc_Segment.save_json(json_file)

dc_Segment_from_json = dataContainer.from_json(json_file)
dc_Segment_from_json.multi_file = ['a', 'b', 'c']


json_file = '/Users/carlos.aguilar/Documents/stupid_test/container_2.json'
dc_Segment_from_json.save_json(json_file)




# create accessor
athena = athenaHandle.getAthenaHandle()


data_type = 'zip'
file_location = 's3://af-datalake/tester/#46.zip'
file_location = '/Users/carlosAguilar/Documents/raw data/Arrixaca_2k18_revisited/Export_Study-1-12_15_2017-11-08-38.zip'
description = 'File from Hospital Arrixaca'
dc_case46 = dataContainer(data_type, file_location, description = description)
dc_case46.to_json()
dc_case46.save_json()




@dataclass
class dataInstance:
    data_type: str
    file_location: str
    creation_time: dt.datetime = dt.datetime.utcnow().\
        replace(tzinfo=dt.timezone.utc).isoformat()
    description: str = 'Not provided'
    data_container: []
    producer: []
    consumer: str
    def to_json(self):
        return json.dumps(asdict(self), indent=4, default=str)
    def save_json(self):
        json_blob = self.to_json()
        # Get the file with json extension
        [fPath, fName] = os.path.split(file_location)
        [file, ext] = os.path.splitext(fName)
        json_file = os.path.join(fPath, file + '.json')
        with open(json_file, 'w+') as in_file:
            in_file.write(json_blob)




@dataclass
class StarWarsMovie:
  title: str
  episode_id: int
  opening_crawl: str
  director: str
  producer: str
  release_date: datetime
  created: datetime
  edited: datetime
  url: str
  characters: List[str]
  planets: List[str]
  starships: List[str]
  vehicles: List[str]
  species: List[str]




@dataclass
class Metadata:
    # Non-nullable Pseudo fields
    author_names: InitVar[list]
    author_ids: InitVar[list]
    # Normal fields
    title: str
    url: str
    created_at: str = None
    authors: list = None
    # Calculated fields
    post_id: str = field(init=False)

    def __post_init__(self, author_names, author_ids): # You have to pass pseudo fields as the parameter.
        random_number = random.randint(100000, 999999)
        self.post_id = f"{random_number}_{self.url}"
        if author_names:
            self.authors = []
            for i in range(0, len(author_names)):
                self.authors.append({"id": author_ids[i], "name": author_names[i]})

    def to_json(self):    
        metadata = asdict(self)
        for key in list(metadata):
            if key != "url" and metadata[key] == None:
                    del metadata[key]
        return json.dumps(metadata)


m = Metadata(title="Some Article", 
url="https://example.com/article.html",
created_at="2018-09-23",
author_names=["Furkan Kalkan", "John Doe"],
author_ids=["1", "2"])

m.to_json()






m1 = Metadata(title="Some Other Article", 
url="https://example.com/article.html",
created_at="2018-09-23",
author_names=["John Doe"],
author_ids=["13"])

# Replace functionality (not in that definition)
# m1 = replace(m1, url="https://example.com/other_article.html")

# Access the annotations
m1.__annotations__



@dataclass
class Person:
  name: str
  age: int
  job: str = "Developer"

samantha = Person("Samantha Carter", 33)

# Replace
samantha = replace(samantha, job="Frontend Developer")
samantha.age += 1

Person.__dataclass_params__



# Use field for values that can be updated at a later time, instead of creation time
@dataclass
class Booker2:
  author  : str = field()
  title   : str = field()
  isbn    : int = field(compare=False)
  price   : int = field(default_factory=int, metadata={"currency":"Turkish Lira"})
  renters : list = field(default_factory=list, metadata={"max": 5}, repr=False)
  def rent(self, name):
    if len(self.renters) >= 5:
        raise ValueError("5 People Already Rent This Book")
    self.renters.append(name)
  def unrent(self, name):
    self.renters.remove(name)

 
free = Booker2("Sam Williams", "Free as in Freedom", 9968237238, 35)
cb   = Booker("Eric Raymond", "Cathedral and the Bazaar", 969398332)
cb.rent('Pepe')



@dataclass
class InventoryItems:
  '''Class for keeping track of an item in inventory.'''
  name: str
  unit_price: float
  quantity_on_hand: int = 0
  item_description: str = 'field()'

  def total_cost(self) -> float:
      return self.unit_price * self.quantity_on_hand




from dataclasses import dataclass
from typing import List

@dataclass
class PlayingCard:
    rank: str
    suit: str

@dataclass
class Deck:
    cards: List[PlayingCard]