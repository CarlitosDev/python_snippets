#lru_cache (memoization).py


# standard
import time
def fib(number: int) -> int:
 if number == 0: return 0
 if number == 1: return 1
return fib(number-1) + fib(number-2)
start = time.time()
 fib(40)
 print(f’Duration: {time.time() — start}s’)
 # Duration: 30.684099674224854s

# using memoization
# LRU (Least Recently Used) cache 
from functools import lru_cache
@lru_cache(maxsize=512)
 def fib_memoization(number: int) -> int:
 if number == 0: return 0
 if number == 1: return 1
return fib_memoization(number-1) + fib_memoization(number-2)
start = time.time()
 fib_memoization(40)
 print(f’Duration: {time.time() — start}s’)
 # Duration: 6.866455078125e-05s