'''
generate_NanoID.py


-NanoID is 60% faster than the UUID. Instead of having 36 characters in UUID’s alphabet, NanoID only has 21characters.
- 
'''


source ~/.bash_profile && python3 -m pip install nanoid


'''
Normal
The main module uses URL-friendly symbols (A-Za-z0-9_-) and returns an ID with 21 characters (to have a collision probability similar to UUID v4).
'''
from nanoid import generate
generate()



'''
If you want to reduce ID length (and increase collisions probability), you can pass the length as an argument.
'''
generate(size=10)


'''
If you want to change the ID's alphabet or length you can use the internal generate module.
'''

generate('abcdef', 10)


