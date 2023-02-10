import psycopg2
from sshtunnel import SSHTunnelForwarder
import utils.data_utils as du
from collections import deque
import pandas as pd



#
d = deque('abcd')

# basic operations
# left
d.appendleft('0')
# right
d.appendleft('e')
# Get position to insert a new object
position_c = d.index('c')
# inser 'before'
d.insert(position_c, 'c1')

for elem in d:
    print(elem.upper())

# Shift positions
e = d.copy()
e.rotate(3)

for elem in e:
    print(elem.upper())



####
# Try queues with dictionaries
a1 = {'level': 'B1', 'last_activity': 5, 'last_score': 9}
d = deque(s1)




###
iterable = [40, 30, 50, 46, 39, 44]
a = moving_average([40, 30, 50, 46, 39, 44])


def moving_average(iterable, n=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # http://en.wikipedia.org/wiki/Moving_average
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n