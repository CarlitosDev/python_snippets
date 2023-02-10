MAPS

# here 'len' is the actual method that we are passing
name_lengths = map(len, ["Mary", "Isla", "Sam"])
print(list(name_lengths))
# or we can pass our preferred method via lamdba
squares = [*map(lambda x: x * x, [0, 1, 2, 3, 4])]

# Replace strings
pickleFileName = [*map(lambda x: x.replace('.csv.gz', '_Summary.pickle'), srcFileList)]

# Retain the second returned value using maps
justFileNames = [*map(lambda x: os.path.split(x)[1],  srcFileList)];


# Apply a function to a list and get the result also as a list:
cleanList = lambda x: str.replace(x, " ", "_").lower()
a = list(map(cleanList, tempColNames))


# Reduce: Reduce takes a function and a collection of items. It returns a value that
# is created by combining the items.
from functools import 
sum = reduce(lambda a, x: a + x, [0, 1, 2, 3, 4])