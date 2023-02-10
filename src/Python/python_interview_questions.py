Interview questions:

(1) Output of this??
def multipliers():
	return [lambda x: i * x for i in range(4)]

print([m(3) for m in multipliers()])

>> Answer: 
Explanation: Four functions are created; instead all of them just multiply x by i.

Python’s closures are late binding, the value of the variable ‘i’ is looked up when 
any of the functions returned by multipliers are called. This means that the values of 
variables used in closures are looked up at the time the inner function is called.

Whenever any of the returned functions are called, the value of i is looked up 
in the surrounding scope at call time. By then, the loop has completed and i is left with its final value of 4.



(2) How can you check if a data set or time series is Random?

The PDF? Kind of...
Kolmogorov-Smirnov test? Kind of...

The answer that I found on the internet is to use 'lag plot', which I believe they mean a proxy for the autocorr.

(3) State the difference between tuples and lists in Python.

Lists: Collections of elements of any type, accessible by index.
Tuples: Sets of elements.
Answers on the web: Tuples can be used a key in a dictionary to store notes on locations 
whereas a list can be used to store multiple locations. Lists are mutable whereas tuples are immutable which means they cannot be edited.

(4) Name a few libraries in Python used for Data Analysis and Scientific computations.
Pandas, statsmodels, numpy, scikit-learn. Computing Dask

(5) Write the code to sort an array in NumPy by the (n-1)th column?
np.sort(array)[::-1]
Web: x[x [: n-2].argsort ()]

(6) If you are to give the first and last names of employees, which data type in Python will you use to store them?
Dictionary
You can use a list that has first name and last name included in an element or use Dictionary.

(7) Explain the usage of decorators.
Wrappers to functions.

On the web: A decorator is a function that takes another function and extends the behavior 
of the latter function without explicitly modifying it. 
They are used to modify the code in classes and functions.
With the help of decorators a piece of code can be executed before or after the execution of the original code.

(8)

def foo (i=[]):
	i.append(1)
	return i
foo()

(9) Which tool in Python would you use to find bugs?
The tools to find bugs in Python are Pylint and Pychecker. 
Pylint is used to verify if a module satisfies all the coding standards. 
Pychecker is a static analysis tool that helps to find out bugs in the source code.

(10) What is the difference between range () and xrange () functions in Python?
The range () function returns a list whereas the xrange () function returns an object 
that works like an iterator for generating numbers on demand.

(11) What will be the output of the below code
word = 'aeioubcdfg'
word[:3]
word[:]
word[-3:-1]
word[1:3]
print(word[:3] + word[:3])


(12 )What is power analysis?
An experimental design technique for determining the effect of a given sample size.