from James Powell - https://www.youtube.com/watch?v=cKPlPJyQrt4

and summarised here: https://medium.com/@mozesr/pydata17-what-does-it-take-to-be-an-expert-at-python-198f19cd2b


Python Data Models

Use the ‘underscore’ functions when writing Classes to compact your code and add modularity.

__init__- object constructor
__add__- addition of objects
__repr__- what to show when string representation of the object is required
__len__- what to return when calling len(self)
__call__ — *()->?

He describes this mechanism very nicely — the underscore function describes a protocol and 
there is always some high level operation that triggers this protocol (new, +, len etc). 
Usually the protocol just delegates the same protocol to the internal components of the class 
— we see this pattern all the time.


class Polynomial:
	def __init__(self, *coeffs):
		self.coeffs = coeffs

	def __repr__(self):
		return 'Polynomial(*{!r})'.format(self.coeffs)

	def __add__(self, other):
		return Polynomial(*(x + y for x,y in zip(self.coeffs, other.coeffs)))

	def __len__(self):
		return len(self.coeffs)


p1 = Polynomial(54,2)
p2 = Polynomial(1,94)

p1.__repr__




