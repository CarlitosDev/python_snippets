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


import sys
assert sys.version_info.major is 3, 'Not running Python 3'
import os
from dataclasses import *
import random


import random, json
from dataclasses import *


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
  def __post_init__(self):
    if type(self.release_date) is str:
    self.release_date = dateutil.parser.parse(self.release_date)
  
    if type(self.created) is str:
        self.created = dateutil.parser.parse(self.created)
  
    if type(self.edited) is str:
        self.edited = dateutil.parser.parse(self.edited)
  # same post init by Peter Norvig     
  def __post_init__(self):
  for attr in [‘release_date’, ‘created’, ‘edited’]:
        if isinstance(getattr(self, attr), str):
              setattr(self, attr, dateutil.parser.parse(getattr(self, attr)))


