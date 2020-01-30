'''
    Important class methods that shouldn't be overlooked
'''

# Get all the methods of a class/object
object_methods = [method_name for method_name in dir(this_object)
                  if callable(getattr(this_object, method_name))]



#  the format code {0.x} specifies the x-attribute of argument 0. So, in the following function, the 0 is actually the instance self
def __repr__(self):
    return 'Pair({0.x!r}, {0.y!r})'.format(self)
def __str__(self):
    return '({0.x!s}, {0.y!s})'.format(self)


# The _format_() method provides a hook into Python’s string formatting functionality.
_formats = {
    'ymd' : '{d.year}-{d.month}-{d.day}',
    'mdy' : '{d.month}/{d.day}/{d.year}',
    'dmy' : '{d.day}/{d.month}/{d.year}'
    }

class Date(object):
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)




'''
    Class example to connect to AWS Athena

'''

from datetime import datetime
import boto3
import pandas as pd
import time
from pyathenajdbc import connect

class athenaHandle:
    
    # Private attributes
    __conn = []
    __name           = ''
    __queryStart     = []
    __queryEnd       = []
    __queryElapsed   = []
    __region_name    = ''
    __s3_staging_dir = ''
    __createdAt      = datetime.now()
    
    # Public attributes
    sqlQuery = ''
    
    def getConnection(self):
            return self.__conn
    
    # Constructor
    def __init__(self, name, region_name = 'eu-central-1'):
        self.__name = name
        self.__region_name = region_name
        
    # Public methods        
    def query(self, sqlQuery):
        self.__queryStart = time.time()
        self.sqlQuery     = sqlQuery;
        print('Querying...', end='')
        df    = pd.read_sql(self.sqlQuery, self.__conn)
        [r,c] = df.shape
        print('{} rows and {} cols retrieved'.format(r,c), end='')
        self.__setElapsed__()
        return df
        
    def getLastQuery(self):
        return self.sqlQuery
        
    # Private methods
    def __setElapsed__(self):
        self.__queryEnd     = time.time()
        self.__queryElapsed = self.__queryEnd - self.__queryStart
        print('...done in {:.2f} sec!'.format(self.__queryElapsed))
        

    def __setConnection__(self):
        credentials = boto3.Session(region_name=self.__region_name).get_credentials()
        conn = connect(access_key=credentials.access_key,
                       secret_key=credentials.secret_key,
                       s3_staging_dir=self.__s3_staging_dir,
                       region_name=self.__region_name)
        self.__conn = conn

    # Class method. Do not use the constructor directly
    @classmethod
    def getAthenaHandle(cls, region_name = 'eu-central-1', \
        s3_staging_dir='s3://aws-athena-query-results-647330586553-eu-central-1/'):
        self = cls(name='AthenaQuery')
        self.__s3_staging_dir = s3_staging_dir
        self.__queryStart = time.time()
        self.__setConnection__()
        self.__setElapsed__()
        print('{} created at {}'.format(self.__name, self.__createdAt))
        return self

    # Class method. Do not use the constructor directly
    @classmethod
    def getS2DSConnector(cls, region_name = 'eu-central-1', \
        s3_staging_dir='s3://aws-athena-query-results-647330586553-eu-central-1/'):
        self = cls(name='AthenaQuery')
        self.__s3_staging_dir = s3_staging_dir
        self.__queryStart = time.time()
        conn = connect(access_key='AKIAJMWJAQI73OP7NNDQ',
                       secret_key='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',
                       s3_staging_dir=self.__s3_staging_dir,
                       region_name=self.__region_name)
        self.__conn = conn
        self.__setElapsed__()
        print('{} created at {}'.format(self.__name, self.__createdAt))
        return self

        



############
# metaclasses
############

class BaseMeta(type):
    # Approach 1
    def __new__(cls, name, bases, body):
        print('BaseMeta.__new__', cls, name, bases, body)
        if name != 'Base' and not 'bar' in body:
            raise TyperError('bad class')
        return super().__new__(cls, name, bases, body)
    # Approach 2 (introduced P3.6)
    

class Base(metaclass=BaseMeta):  
    def foo(self):
        return self.bar()
    def __init_subclass__(self, *a, **kw):
        print('init_subclass', a, kw)
        return super().__init_subclass__(*a, **kw)


class Derived(Base):
    def bar(self):
        return 'bar'
